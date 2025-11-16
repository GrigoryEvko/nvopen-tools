// Function: sub_B101E0
// Address: 0xb101e0
//
__int64 __fastcall sub_B101E0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 *v13; // rcx
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  unsigned int v16; // r12d
  unsigned int v17; // edx
  __int64 *v18; // r8
  __int64 v19; // rcx
  int v20; // r9d
  __int64 *v21; // r10
  int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h] BYREF
  __int64 v26[6]; // [rsp+20h] [rbp-30h] BYREF

  v23 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v23;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v24 = 0;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v11 = a1 - 16;
  LODWORD(v24) = *(unsigned __int16 *)(a1 + 2);
  HIDWORD(v24) = *(_DWORD *)(a1 + 4);
  v12 = *(_BYTE *)(a1 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(__int64 **)(a1 - 32);
  else
    v13 = (__int64 *)(v11 - 8LL * ((v12 >> 2) & 0xF));
  v25 = *v13;
  v14 = *(_BYTE *)(a1 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(a1 - 32);
  else
    v15 = v11 - 8LL * ((v14 >> 2) & 0xF);
  v16 = v4 - 1;
  v26[0] = *(_QWORD *)(v15 + 8);
  v8 = v23;
  v17 = v16 & sub_AFBBF0((int *)&v24, (int *)&v24 + 1, &v25, v26);
  v18 = (__int64 *)(v10 + 8LL * v17);
  result = v23;
  v19 = *v18;
  if ( *v18 == v23 )
    return result;
  v20 = 1;
  v7 = 0;
  while ( v19 != -4096 )
  {
    if ( v19 != -8192 || v7 )
      v18 = v7;
    v17 = v16 & (v20 + v17);
    v21 = (__int64 *)(v10 + 8LL * v17);
    v19 = *v21;
    if ( *v21 == v23 )
      return result;
    ++v20;
    v7 = v18;
    v18 = (__int64 *)(v10 + 8LL * v17);
  }
  v22 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v18;
  ++*(_QWORD *)a3;
  v9 = v22 + 1;
  v24 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B0FF40(a3, v6);
  sub_AFF920(a3, &v23, &v24);
  v7 = v24;
  v8 = v23;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v23;
}
