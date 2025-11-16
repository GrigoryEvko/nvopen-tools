// Function: sub_1E9B7B0
// Address: 0x1e9b7b0
//
_BOOL8 __fastcall sub_1E9B7B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 i; // r15
  unsigned __int16 v4; // cx
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  const __m128i *v13; // rax
  const __m128i *v14; // r14
  const __m128i *v15; // rdx
  bool v17; // [rsp+7h] [rbp-69h]
  const __m128i *v18; // [rsp+8h] [rbp-68h]
  __m128i v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 v21; // [rsp+28h] [rbp-48h]

  v17 = sub_15602E0((_QWORD *)(*(_QWORD *)a2 + 112LL), "patchable-function", 0x12u);
  if ( v17 )
  {
    v2 = *(__int64 **)(a2 + 328);
    for ( i = v2[4]; ; i = *(_QWORD *)(i + 8) )
    {
      v4 = **(_WORD **)(i + 16);
      if ( v4 > 0xDu || ((1LL << v4) & 0x325C) == 0 )
        break;
      if ( (*(_BYTE *)i & 4) == 0 && (*(_BYTE *)(i + 46) & 8) != 0 )
      {
        do
          i = *(_QWORD *)(i + 8);
        while ( (*(_BYTE *)(i + 46) & 8) != 0 );
      }
    }
    v5 = *(_QWORD *)(a2 + 16);
    v6 = *(__int64 (**)())(*(_QWORD *)v5 + 40LL);
    if ( v6 == sub_1D00B00 )
      BUG();
    v7 = ((__int64 (__fastcall *)(__int64, const char *, __int64))v6)(v5, "patchable-function", 1);
    v8 = v2[7];
    v9 = (__int64)sub_1E0B640(v8, *(_QWORD *)(v7 + 8) + 1664LL, (__int64 *)(i + 64), 0);
    sub_1DD5BA0(v2 + 2, v9);
    v10 = *(_QWORD *)i;
    v11 = *(_QWORD *)v9;
    *(_QWORD *)(v9 + 8) = i;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v9 = v10 | v11 & 7;
    *(_QWORD *)(v10 + 8) = v9;
    *(_QWORD *)i = v9 | *(_QWORD *)i & 7LL;
    v19.m128i_i64[0] = 1;
    v20 = 0;
    v21 = 2;
    sub_1E1A9C0(v9, v8, &v19);
    v12 = **(unsigned __int16 **)(i + 16);
    v19.m128i_i64[0] = 1;
    v20 = 0;
    v21 = v12;
    sub_1E1A9C0(v9, v8, &v19);
    v13 = *(const __m128i **)(i + 32);
    v18 = (const __m128i *)((char *)v13 + 40 * *(unsigned int *)(i + 40));
    if ( v13 != v18 )
    {
      v14 = *(const __m128i **)(i + 32);
      do
      {
        v15 = v14;
        v14 = (const __m128i *)((char *)v14 + 40);
        sub_1E1A9C0(v9, v8, v15);
      }
      while ( v18 != v14 );
    }
    sub_1E16240(i);
    if ( *(_DWORD *)(a2 + 340) <= 3u )
      *(_DWORD *)(a2 + 340) = 4;
  }
  return v17;
}
