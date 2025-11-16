// Function: sub_26A4400
// Address: 0x26a4400
//
__int64 __fastcall sub_26A4400(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // r13
  _QWORD *v10; // r15
  char v12; // [rsp+Fh] [rbp-71h] BYREF
  int v13; // [rsp+10h] [rbp-70h] BYREF
  int v14; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h] BYREF
  __m128i v16; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v17; // [rsp+30h] [rbp-50h] BYREF
  int *v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]
  __m128i *v20; // [rsp+48h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 72);
  v4 = v3 & 3;
  v5 = v3 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v4 == 3 )
    v5 = *(_QWORD *)(v5 + 24);
  v6 = *(_BYTE *)v5;
  if ( *(_BYTE *)v5 )
  {
    if ( v6 == 22 )
    {
      v5 = *(_QWORD *)(v5 + 24);
    }
    else if ( v6 <= 0x1Cu )
    {
      v5 = 0;
    }
    else
    {
      v5 = sub_B43CB0(v5);
    }
  }
  v18 = 0;
  v17 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFFCLL);
  nullsub_1518();
  v7 = sub_26A2E60(a2, v5 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 0);
  v8 = *(_BYTE *)(v7 + 97) == 0;
  v15 = v7;
  if ( v8 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  v13 = *(_DWORD *)(a1 + 100);
  v9 = v13;
  v10 = (_QWORD *)(a1 + 16LL * v13 + 104);
  v19 = a2;
  v17 = &v15;
  v18 = &v13;
  v20 = &v16;
  v12 = 0;
  v14 = 1;
  v16 = 0;
  if ( (unsigned __int8)sub_2526370(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_266DE70,
                          (__int64)&v17,
                          a1,
                          &v14,
                          1,
                          &v12,
                          1,
                          0) )
  {
    if ( v16.m128i_i8[8] != *(_BYTE *)(a1 + 16 * v9 + 112) )
    {
LABEL_10:
      *(__m128i *)(a1 + 16 * v9 + 104) = _mm_loadu_si128(&v16);
      return 0;
    }
    if ( !v16.m128i_i8[8] )
      return 1;
  }
  else
  {
    v16.m128i_i64[0] = 0;
    if ( v16.m128i_i8[8] )
    {
      if ( !*(_BYTE *)(a1 + 16 * v9 + 112) )
        goto LABEL_10;
    }
    else
    {
      v8 = *(_BYTE *)(a1 + 16 * v9 + 112) == 0;
      v16.m128i_i8[8] = 1;
      if ( v8 )
        goto LABEL_10;
    }
  }
  if ( v16.m128i_i64[0] != *v10 )
    goto LABEL_10;
  return 1;
}
