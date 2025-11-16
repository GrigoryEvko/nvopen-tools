// Function: sub_1466F20
// Address: 0x1466f20
//
__int64 __fastcall sub_1466F20(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r8
  unsigned int v8; // r13d
  int v10; // edx
  unsigned int v11; // esi
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  int v16; // r11d
  __int64 *v17; // r8
  int v18; // eax
  int v19; // edx
  int v20; // r9d
  __int64 *v21; // [rsp+8h] [rbp-48h] BYREF
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  bool (__fastcall *v23)(__int64); // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 104);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 88);
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
        return *((unsigned __int8 *)v6 + 8);
    }
    else
    {
      v10 = 1;
      while ( v7 != -8 )
      {
        v20 = v10 + 1;
        v5 = (v3 - 1) & (v10 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v10 = v20;
      }
    }
  }
  LOBYTE(v22) = 0;
  v23 = sub_14525E0;
  sub_145E0E0(a2, (__int64)&v22);
  v8 = (unsigned __int8)v22;
  v11 = *(_DWORD *)(a1 + 104);
  v22 = a2;
  LOBYTE(v23) = v8;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 80);
LABEL_22:
    v11 *= 2;
    goto LABEL_23;
  }
  v12 = *(_QWORD *)(a1 + 88);
  v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( a2 == *v14 )
    return v8;
  v16 = 1;
  v17 = 0;
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v17 )
      v17 = v14;
    v13 = (v11 - 1) & (v16 + v13);
    v14 = (__int64 *)(v12 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
      return v8;
    ++v16;
  }
  if ( !v17 )
    v17 = v14;
  v18 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 80);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v11 )
    goto LABEL_22;
  if ( v11 - *(_DWORD *)(a1 + 100) - v19 <= v11 >> 3 )
  {
LABEL_23:
    sub_1466D60(a1 + 80, v11);
    sub_145F580(a1 + 80, &v22, &v21);
    v17 = v21;
    a2 = v22;
    v19 = *(_DWORD *)(a1 + 96) + 1;
  }
  *(_DWORD *)(a1 + 96) = v19;
  if ( *v17 != -8 )
    --*(_DWORD *)(a1 + 100);
  *v17 = a2;
  *((_BYTE *)v17 + 8) = (_BYTE)v23;
  return v8;
}
