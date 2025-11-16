// Function: sub_89F220
// Address: 0x89f220
//
__int64 ***__fastcall sub_89F220(__int64 *a1, const __m128i *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rbx
  __m128i *i; // rax
  __m128i *v13; // r15
  __int64 v14; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 **v18; // [rsp+8h] [rbp-48h]
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a3 + 88);
  v5 = *a1;
  v19[0] = v4;
  v18 = qword_4D03B88;
  if ( *(char *)(v5 + 130) >= 0 && !*(_QWORD *)(v4 + 240) )
    qword_4D03B88 = 0;
  sub_89C1A0((__int64)a1, a3, a2, v19, (FILE *)(a3 + 48));
  if ( a1[61] )
    goto LABEL_5;
  v16 = sub_727300();
  v6 = qword_4F04C68;
  v17 = v16;
  v16[3] = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
  v16[4] = *(_QWORD *)&dword_4F077C8;
  *v16 = a1[61];
  v9 = a1[24];
  a1[61] = (__int64)v17;
  if ( v9 )
  {
    *(_QWORD *)(v9 + 32) = v17;
LABEL_5:
    v9 = a1[24];
  }
  v10 = *(_QWORD **)v9;
  v11 = *(_QWORD *)(v9 + 32);
  for ( i = 0; v10; v10 = (_QWORD *)*v10 )
  {
    while ( 1 )
    {
      v13 = i;
      i = sub_8992B0((__int64)v10);
      if ( !v13 )
        break;
      v13[7].m128i_i64[0] = (__int64)i;
      v10 = (_QWORD *)*v10;
      if ( !v10 )
        goto LABEL_11;
    }
    *(_QWORD *)(v11 + 8) = i;
  }
LABEL_11:
  v14 = a1[61];
  *(_QWORD *)(a1[42] + 176) = v14;
  sub_8911B0((__int64)a1, a3, v14, (__int64)v6, v7, v8);
  qword_4D03B88 = v18;
  return &qword_4D03B88;
}
