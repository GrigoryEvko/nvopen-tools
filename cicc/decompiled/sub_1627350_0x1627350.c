// Function: sub_1627350
// Address: 0x1627350
//
__int64 __fastcall sub_1627350(__int64 *a1, __int64 *a2, __int64 *a3, int a4, char a5)
{
  __int64 *v5; // r9
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 *v17; // rcx
  __int64 v18; // rax
  int v19; // r10d
  __int64 *v20; // rdx
  int v21; // ecx
  int v22; // eax
  __int64 v25; // [rsp+10h] [rbp-80h]
  int v26; // [rsp+1Ch] [rbp-74h]
  __int64 v27; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v28[4]; // [rsp+30h] [rbp-60h] BYREF
  int v29; // [rsp+50h] [rbp-40h]

  v5 = a3;
  if ( a4 )
  {
    v26 = 0;
    v9 = *a1;
  }
  else
  {
    v28[0] = a2;
    v28[1] = a3;
    v28[2] = 0;
    v28[3] = 0;
    v8 = sub_1607C40(a2, (__int64)a3);
    v9 = *a1;
    v26 = v8;
    v10 = *a1 + 496;
    v29 = v8;
    v11 = sub_161C9B0(v10, (__int64)v28);
    if ( v11 )
      return v11;
    v5 = a3;
    if ( !a5 )
      return v11;
  }
  v25 = (__int64)v5;
  v13 = sub_161E980(24, (unsigned int)v5);
  v11 = v13;
  if ( v13 )
  {
    sub_1623D80(v13, (__int64)a1, 4, a4, a2, v25, 0, 0);
    *(_DWORD *)(v11 + 4) = v26;
  }
  v27 = v11;
  if ( a4 )
  {
    if ( a4 == 1 )
    {
      sub_1621390((char *)v11);
      return v27;
    }
    return v11;
  }
  v14 = *(_DWORD *)(v9 + 520);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v9 + 504);
    v16 = (v14 - 1) & *(_DWORD *)(v11 + 4);
    v17 = (__int64 *)(v15 + 8LL * v16);
    v18 = *v17;
    if ( v11 == *v17 )
      return v11;
    v19 = 1;
    v20 = 0;
    while ( v18 != -8 )
    {
      if ( v18 != -16 || v20 )
        v17 = v20;
      v16 = (v14 - 1) & (v19 + v16);
      v18 = *(_QWORD *)(v15 + 8LL * v16);
      if ( v11 == v18 )
        return v11;
      ++v19;
      v20 = v17;
      v17 = (__int64 *)(v15 + 8LL * v16);
    }
    v22 = *(_DWORD *)(v9 + 512);
    if ( !v20 )
      v20 = v17;
    ++*(_QWORD *)(v9 + 496);
    v21 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v14 )
    {
      if ( v14 - *(_DWORD *)(v9 + 516) - v21 > v14 >> 3 )
        goto LABEL_23;
      goto LABEL_22;
    }
  }
  else
  {
    ++*(_QWORD *)(v9 + 496);
  }
  v14 *= 2;
LABEL_22:
  sub_1627160(v9 + 496, v14);
  sub_1621680(v9 + 496, &v27, v28);
  v20 = v28[0];
  v11 = v27;
  v21 = *(_DWORD *)(v9 + 512) + 1;
LABEL_23:
  *(_DWORD *)(v9 + 512) = v21;
  if ( *v20 != -8 )
    --*(_DWORD *)(v9 + 516);
  *v20 = v11;
  return v27;
}
