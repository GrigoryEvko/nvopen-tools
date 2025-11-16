// Function: sub_26A4090
// Address: 0x26a4090
//
__int64 __fastcall sub_26A4090(__m128i *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  _BYTE *v7; // rax
  _BYTE *v8; // r13
  __int64 result; // rax
  __int64 (__fastcall *v10)(_BYTE *, _QWORD, unsigned __int64, __int64); // r15
  unsigned __int64 v11; // r14
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  __int8 v15; // dl
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r14
  __m128i v20; // [rsp+0h] [rbp-40h] BYREF

  v3 = a1[4].m128i_i64[1];
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
  v20 = (__m128i)(v5 & 0xFFFFFFFFFFFFFFFCLL);
  nullsub_1518();
  v7 = (_BYTE *)sub_26A2E60(a2, v5 & 0xFFFFFFFFFFFFFFFCLL, 0, (__int64)a1, 0);
  v8 = v7;
  if ( !v7[97] )
  {
    a1[6].m128i_i8[1] = a1[6].m128i_i8[0];
    return 0;
  }
  v10 = *(__int64 (__fastcall **)(_BYTE *, _QWORD, unsigned __int64, __int64))(*(_QWORD *)v7 + 112LL);
  v11 = a1[4].m128i_i64[1] & 0xFFFFFFFFFFFFFFFCLL;
  if ( (a1[4].m128i_i64[1] & 3) == 3 )
  {
    v11 = *(_QWORD *)(v11 + 24);
    v12 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 > 0x1Cu )
      goto LABEL_12;
LABEL_21:
    if ( v12 == 22 )
    {
      if ( !sub_B2FC80(*(_QWORD *)(v11 + 24)) )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 80LL);
        if ( !v16 )
          BUG();
        v17 = *(_QWORD *)(v16 + 32);
        v11 = v17 - 24;
        if ( v17 )
          goto LABEL_12;
LABEL_25:
        v11 = 0;
        goto LABEL_12;
      }
      v12 = *(_BYTE *)v11;
    }
    if ( !v12 && !sub_B2FC80(v11) )
    {
      v18 = *(_QWORD *)(v11 + 80);
      if ( !v18 )
        BUG();
      v19 = *(_QWORD *)(v18 + 32);
      if ( v19 )
      {
        v11 = v19 - 24;
        goto LABEL_12;
      }
    }
    goto LABEL_25;
  }
  v12 = *(_BYTE *)v11;
  if ( *(_BYTE *)v11 <= 0x1Cu )
    goto LABEL_21;
LABEL_12:
  v13 = v10(v8, a1[6].m128i_u32[2], v11, a2);
  v20.m128i_i64[1] = v14;
  v15 = a1[7].m128i_i8[8];
  v20.m128i_i64[0] = v13;
  if ( v15 != v20.m128i_i8[8] || (result = 1, v15) && a1[7].m128i_i64[0] != v20.m128i_i64[0] )
  {
    a1[7] = _mm_loadu_si128(&v20);
    return 0;
  }
  return result;
}
