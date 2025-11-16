// Function: sub_32611B0
// Address: 0x32611b0
//
__int64 __fastcall sub_32611B0(
        unsigned int a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v8; // r11d
  __int64 v12; // r10
  __int64 (*v13)(); // rbx
  unsigned __int16 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rbx
  __int64 v19; // r8
  const void *v20; // rsi
  __int64 v21; // r12
  __int64 v22; // rcx
  int v23; // edi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned int v28; // [rsp+0h] [rbp-40h]
  unsigned __int8 v29; // [rsp+6h] [rbp-3Ah]
  unsigned __int8 v30; // [rsp+7h] [rbp-39h]
  unsigned int v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+50h] [rbp+10h]

  v12 = a7;
  v13 = *(__int64 (**)())(*(_QWORD *)a8 + 1392LL);
  v14 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * a5);
  v15 = *((_QWORD *)v14 + 1);
  v16 = *v14;
  v17 = 0;
  if ( v13 != sub_2FE3480 )
  {
    v31 = a6;
    v17 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64))v13)(a8, a1, a2, v16, v15);
    v12 = a7;
    a6 = v31;
  }
  v18 = *(_QWORD *)(a4 + 56);
  v19 = 0;
  v20 = (const void *)(v12 + 16);
  if ( !v18 )
    return 1;
  do
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(v18 + 16);
      if ( v21 == a3 || *(_DWORD *)(v18 + 8) != a5 )
        goto LABEL_8;
      if ( (_DWORD)a6 == 215 || *(_DWORD *)(v21 + 24) != 208 )
      {
        if ( !(_BYTE)v17 )
          return 0;
        if ( *(_DWORD *)(v21 + 24) == 49 )
          v19 = v17;
        goto LABEL_8;
      }
      v22 = *(_QWORD *)(v21 + 40);
      if ( (_DWORD)a6 == 214 && (unsigned int)(*(_DWORD *)(*(_QWORD *)(v22 + 80) + 96LL) - 18) <= 3 )
        return 0;
      LOBYTE(v8) = *(_DWORD *)(v22 + 8) == a5 && *(_QWORD *)v22 == a4;
      if ( !(_BYTE)v8 )
        break;
      v25 = *(_QWORD *)(v22 + 40);
      if ( *(_DWORD *)(v22 + 48) != a5 || v25 != a4 )
        goto LABEL_20;
LABEL_8:
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
        goto LABEL_24;
    }
    v23 = *(_DWORD *)(*(_QWORD *)v22 + 24LL);
    if ( v23 != 11 && v23 != 35 )
      return v8;
    v25 = *(_QWORD *)(v22 + 40);
    if ( *(_DWORD *)(v22 + 48) != a5 || v25 != a4 )
    {
LABEL_20:
      LOBYTE(v25) = *(_DWORD *)(v25 + 24) == 35 || *(_DWORD *)(v25 + 24) == 11;
      v8 = v25;
      if ( !(_BYTE)v25 )
        return v8;
    }
    v26 = *(unsigned int *)(v12 + 8);
    v8 = v26 + 1;
    if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
    {
      v28 = a6;
      v29 = v19;
      v30 = v17;
      v32 = v12;
      sub_C8D5F0(v12, v20, v26 + 1, 8u, v19, a6);
      v12 = v32;
      a6 = v28;
      v19 = v29;
      v17 = v30;
      v26 = *(unsigned int *)(v32 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v12 + 8 * v26) = v21;
    ++*(_DWORD *)(v12 + 8);
    v18 = *(_QWORD *)(v18 + 32);
  }
  while ( v18 );
LABEL_24:
  if ( !(_BYTE)v19 )
    return 1;
  v27 = *(_QWORD *)(a3 + 56);
  if ( !v27 )
    return 1;
  while ( *(_DWORD *)(v27 + 8) || *(_DWORD *)(*(_QWORD *)(v27 + 16) + 24LL) != 49 )
  {
    v27 = *(_QWORD *)(v27 + 32);
    if ( !v27 )
      return 1;
  }
  LOBYTE(v8) = *(_DWORD *)(v12 + 8) != 0;
  return v8;
}
