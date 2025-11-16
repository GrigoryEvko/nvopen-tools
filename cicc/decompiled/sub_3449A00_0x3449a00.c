// Function: sub_3449A00
// Address: 0x3449a00
//
_QWORD *__fastcall sub_3449A00(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __m128i a7,
        __int64 a8,
        unsigned __int64 a9,
        char *a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v15; // r9
  unsigned __int64 v16; // r15
  const void *v17; // rsi
  __int64 (*v18)(); // rax
  char v20; // al
  __int64 v21; // rcx
  __int64 v22; // rax
  int *v23; // rax
  int v24; // edx
  int v25; // ecx
  bool v26; // cc
  int v27; // edx
  __int64 (*v28)(); // rax
  char v29; // al
  __int64 v30; // [rsp+8h] [rbp-78h]

  v15 = a8;
  v16 = a9;
  v17 = a10;
  v18 = *(__int64 (**)())(*(_QWORD *)a1 + 624LL);
  if ( v18 == sub_2FE3180 )
    return sub_33FCE10(a12, a2, a3, a4, a5, a6, a7, v15, v16, v17, a11);
  v20 = ((__int64 (__fastcall *)(__int64, char *, __int64, _QWORD, __int64))v18)(a1, a10, a11, a2, a3);
  v17 = a10;
  v15 = a8;
  if ( v20 )
    return sub_33FCE10(a12, a2, a3, a4, a5, a6, a7, v15, v16, v17, a11);
  v21 = a5;
  a5 = a8;
  v22 = (unsigned int)a6;
  a6 = (unsigned int)a9 | a6 & 0xFFFFFFFF00000000LL;
  v16 = v22 | a9 & 0xFFFFFFFF00000000LL;
  v15 = v21;
  if ( (_DWORD)a11 )
  {
    v23 = (int *)a10;
    do
    {
      v24 = *v23;
      if ( *v23 >= 0 )
      {
        v25 = v24 + a11;
        v26 = v24 < (int)a11;
        v27 = v24 - a11;
        if ( v26 )
          v27 = v25;
        *v23 = v27;
      }
      ++v23;
    }
    while ( v23 != (int *)&a10[4 * (unsigned int)(a11 - 1) + 4] );
  }
  v28 = *(__int64 (**)())(*(_QWORD *)a1 + 624LL);
  if ( v28 == sub_2FE3180 )
    return sub_33FCE10(a12, a2, a3, a4, a5, a6, a7, v15, v16, v17, a11);
  v30 = v15;
  v29 = ((__int64 (__fastcall *)(__int64, char *, __int64, _QWORD, __int64))v28)(a1, a10, a11, a2, a3);
  v15 = v30;
  if ( v29 )
    return sub_33FCE10(a12, a2, a3, a4, a5, a6, a7, v15, v16, v17, a11);
  else
    return 0;
}
