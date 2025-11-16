// Function: sub_1ACCFB0
// Address: 0x1accfb0
//
__int64 __fastcall sub_1ACCFB0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned int v7; // r12d
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // eax
  char v14; // al
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rbx
  __int64 v18; // r13
  __int64 i; // r14
  __int64 v20; // r12
  __int64 v21; // rax
  int v22; // esi
  int v23; // edx
  int v24; // esi
  unsigned __int64 v25; // rbx
  __int64 j; // rcx
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  int v36; // [rsp+18h] [rbp-38h]

  *a4 = 1;
  v7 = sub_1ACCBA0((__int64)a1, a2, a3);
  if ( v7 )
    return v7;
  v7 = sub_1ACA9E0(
         (__int64)a1,
         (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24,
         (unsigned int)*(unsigned __int8 *)(a3 + 16) - 24);
  if ( v7 )
    return v7;
  if ( *(_BYTE *)(a2 + 16) == 56 )
  {
    *a4 = 0;
    v7 = sub_1ACCBA0(
           (__int64)a1,
           *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
           *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    if ( !v7 )
      return sub_1ACCCE0(a1, a2, a3);
    return v7;
  }
  v7 = sub_1ACA9E0((__int64)a1, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, *(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  if ( v7 )
    return v7;
  v7 = sub_1ACB220(a1, *(_QWORD *)a2, *(_QWORD *)a3);
  if ( v7 )
    return v7;
  v7 = sub_1ACA9E0((__int64)a1, *(_BYTE *)(a2 + 17) >> 1, *(_BYTE *)(a3 + 17) >> 1);
  if ( v7 )
    return v7;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v9 = 0;
    v33 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      v10 = (*(_BYTE *)(a3 + 23) & 0x40) != 0 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
      v11 = **(_QWORD **)(v10 + v9);
      v12 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v13 = sub_1ACB220(a1, **(_QWORD **)(v12 + v9), v11);
      if ( v13 )
        return v13;
      v9 += 24;
    }
    while ( v33 != v9 );
  }
  v14 = *(_BYTE *)(a2 + 16);
  switch ( v14 )
  {
    case '5':
      v7 = sub_1ACB220(a1, *(_QWORD *)(a2 + 56), *(_QWORD *)(a3 + 56));
      if ( v7 )
        return v7;
      v15 = (unsigned int)(1 << *(_WORD *)(a3 + 18)) >> 1;
      v16 = (unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1;
      return sub_1ACA9E0((__int64)a1, v16, v15);
    case '6':
      v7 = sub_1ACA9E0((__int64)a1, *(_WORD *)(a2 + 18) & 1, *(_WORD *)(a3 + 18) & 1);
      if ( v7 )
        return v7;
      v7 = sub_1ACA9E0(
             (__int64)a1,
             (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1),
             (unsigned int)(1 << (*(unsigned __int16 *)(a3 + 18) >> 1) >> 1));
      if ( v7 )
        return v7;
      v7 = sub_1ACAA00(
             (__int64)a1,
             (*(unsigned __int16 *)(a2 + 18) >> 7) & 7,
             (*(unsigned __int16 *)(a3 + 18) >> 7) & 7);
      if ( v7 )
        return v7;
      v7 = sub_1ACA9E0((__int64)a1, *(unsigned __int8 *)(a2 + 56), *(unsigned __int8 *)(a3 + 56));
      if ( v7 )
        return v7;
      goto LABEL_37;
    case '7':
      v7 = sub_1ACA9E0((__int64)a1, *(_WORD *)(a2 + 18) & 1, *(_WORD *)(a3 + 18) & 1);
      if ( v7 )
        return v7;
      v7 = sub_1ACA9E0(
             (__int64)a1,
             (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1),
             (unsigned int)(1 << (*(unsigned __int16 *)(a3 + 18) >> 1) >> 1));
      if ( v7 )
        return v7;
      v22 = *(unsigned __int16 *)(a2 + 18) >> 7;
      v23 = (*(unsigned __int16 *)(a3 + 18) >> 7) & 7;
      goto LABEL_42;
  }
  if ( (unsigned __int8)(v14 - 75) <= 1u )
  {
    v15 = *(_WORD *)(a3 + 18) & 0x7FFF;
    v16 = *(_WORD *)(a2 + 18) & 0x7FFF;
    return sub_1ACA9E0((__int64)a1, v16, v15);
  }
  if ( v14 != 78 && v14 != 29 )
  {
    if ( v14 == 87 )
    {
      v17 = *(unsigned int *)(a2 + 64);
      v18 = *(_QWORD *)(a3 + 56);
      v34 = *(_QWORD *)(a2 + 56);
      v7 = sub_1ACA9E0((__int64)a1, v17, *(unsigned int *)(a3 + 64));
      if ( !v7 )
      {
        for ( i = 0; v17 != i; ++i )
        {
          v13 = sub_1ACA9E0((__int64)a1, *(unsigned int *)(v34 + 4 * i), *(unsigned int *)(v18 + 4 * i));
          if ( v13 )
            return v13;
        }
      }
      return v7;
    }
    if ( v14 == 86 )
    {
      v25 = *(unsigned int *)(a2 + 64);
      v35 = *(_QWORD *)(a2 + 56);
      v31 = *(_QWORD *)(a3 + 56);
      v13 = sub_1ACA9E0((__int64)a1, v25, *(unsigned int *)(a3 + 64));
      if ( v13 )
        return v13;
      for ( j = 0; v25 != j; j = v30 + 1 )
      {
        v30 = j;
        v13 = sub_1ACA9E0((__int64)a1, *(unsigned int *)(v35 + 4 * j), *(unsigned int *)(v31 + 4 * j));
        if ( v13 )
          return v13;
      }
      v14 = *(_BYTE *)(a2 + 16);
    }
    if ( v14 == 57 )
    {
      v23 = (*(unsigned __int16 *)(a3 + 18) >> 1) & 0x7FFFBFFF;
      v24 = (*(unsigned __int16 *)(a2 + 18) >> 1) & 0x7FFFBFFF;
      goto LABEL_43;
    }
    if ( v14 == 58 )
    {
      v7 = sub_1ACA9E0((__int64)a1, *(_WORD *)(a2 + 18) & 1, *(_WORD *)(a3 + 18) & 1);
      if ( v7 )
        return v7;
      v7 = sub_1ACA9E0((__int64)a1, *(_BYTE *)(a2 + 19) & 1, *(_BYTE *)(a3 + 19) & 1);
      if ( v7 )
        return v7;
      v7 = sub_1ACAA00(
             (__int64)a1,
             (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
             (*(unsigned __int16 *)(a3 + 18) >> 2) & 7);
      if ( v7 )
        return v7;
      v22 = *(unsigned __int16 *)(a2 + 18) >> 5;
      v23 = (*(unsigned __int16 *)(a3 + 18) >> 5) & 7;
    }
    else
    {
      if ( v14 != 59 )
      {
        if ( v14 == 77 )
        {
          v27 = 0;
          v36 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          while ( 1 )
          {
            v32 = v27;
            if ( v36 == (_DWORD)v27 )
              break;
            v28 = sub_193FF80(a3);
            v29 = sub_193FF80(a2);
            v13 = sub_1ACCBA0((__int64)a1, *(_QWORD *)(v29 + 8 * v32), *(_QWORD *)(v28 + 8 * v32));
            v27 = v32 + 1;
            if ( v13 )
              return v13;
          }
        }
        return v7;
      }
      v7 = sub_1ACA9E0(
             (__int64)a1,
             (*(unsigned __int16 *)(a2 + 18) >> 5) & 0x3FF,
             (*(unsigned __int16 *)(a3 + 18) >> 5) & 0x3FF);
      if ( v7 )
        return v7;
      v7 = sub_1ACA9E0((__int64)a1, *(_WORD *)(a2 + 18) & 1, *(_WORD *)(a3 + 18) & 1);
      if ( v7 )
        return v7;
      v22 = *(unsigned __int16 *)(a2 + 18) >> 2;
      v23 = (*(unsigned __int16 *)(a3 + 18) >> 2) & 7;
    }
LABEL_42:
    v24 = v22 & 7;
LABEL_43:
    v7 = sub_1ACAA00((__int64)a1, v24, v23);
    if ( v7 )
      return v7;
    v15 = *(unsigned __int8 *)(a3 + 56);
    v16 = *(unsigned __int8 *)(a2 + 56);
    return sub_1ACA9E0((__int64)a1, v16, v15);
  }
  v7 = sub_1ACA9E0(
         (__int64)a1,
         (*(unsigned __int16 *)(a2 + 18) >> 2) & 0x1FFF,
         (*(unsigned __int16 *)(a3 + 18) >> 2) & 0x1FFF);
  if ( v7 )
    return v7;
  v7 = sub_1ACAC80((__int64)a1, *(_QWORD *)(a2 + 56), *(_QWORD *)(a3 + 56));
  if ( v7 )
    return v7;
  v7 = sub_1ACAEF0((__int64)a1, a2, a3);
  if ( v7 )
    return v7;
LABEL_37:
  v20 = sub_13CF9A0(a3, 4);
  v21 = sub_13CF9A0(a2, 4);
  return sub_1ACAE20((__int64)a1, v21, v20);
}
