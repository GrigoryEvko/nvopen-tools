// Function: sub_7029D0
// Address: 0x7029d0
//
__int64 __fastcall sub_7029D0(__int64 *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdi
  __int64 v11; // r8
  char v12; // al
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  char v19; // r15
  __int64 v20; // rax
  char i; // dl
  __int64 v23; // rax
  char v24; // r15
  __int64 v25; // [rsp+8h] [rbp-1F8h]
  __int64 v26; // [rsp+10h] [rbp-1F0h]
  __int64 v27; // [rsp+18h] [rbp-1E8h] BYREF
  _QWORD *v28; // [rsp+20h] [rbp-1E0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-1D8h] BYREF
  _QWORD v30[2]; // [rsp+30h] [rbp-1D0h] BYREF
  __m128i v31; // [rsp+40h] [rbp-1C0h]
  __m128i v32; // [rsp+50h] [rbp-1B0h]
  __m128i v33; // [rsp+60h] [rbp-1A0h]
  __m128i v34; // [rsp+70h] [rbp-190h] BYREF
  char v35; // [rsp+82h] [rbp-17Eh]

  v10 = *a1;
  v27 = a4;
  v11 = sub_8D2220(v10);
  v12 = *(_BYTE *)(v11 + 140);
  if ( v12 == 12 )
  {
    v11 = sub_7CFE40(*a1);
  }
  else
  {
    if ( v12 != 14 )
    {
      if ( (unsigned __int8)(v12 - 9) <= 2u )
        goto LABEL_4;
LABEL_15:
      if ( (unsigned int)sub_6E5430() )
        sub_685360(0x99u, (_DWORD *)a1 + 17, *a1);
      goto LABEL_17;
    }
    v11 = sub_7CFE40(v11);
  }
  if ( !v11 )
    goto LABEL_15;
LABEL_4:
  v25 = v11;
  v13 = *(__int64 *)((char *)a1 + 68);
  v30[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v31 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v32 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v33 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v30[1] = v13;
  v14 = strlen(a2);
  sub_878540(a2, v14);
  v15 = sub_7D2AC0(v30, v25, 0);
  v16 = v15;
  if ( !v15 )
    goto LABEL_28;
  v17 = *(unsigned __int8 *)(v15 + 80);
  if ( (*(_BYTE *)(v16 + 81) & 0x10) != 0 )
  {
    if ( (unsigned __int8)v17 > 0x14u )
      goto LABEL_28;
    v18 = 1180672;
    if ( _bittest64(&v18, v17) )
    {
      if ( v31.m128i_i64[1]
        && (unsigned int)sub_84C4B0(
                           v31.m128i_i32[2],
                           a3 != 0,
                           a3,
                           1,
                           (_DWORD)a1,
                           (unsigned int)&v27,
                           0,
                           0,
                           0,
                           0,
                           0,
                           1,
                           6,
                           0,
                           (__int64)a1 + 68,
                           0,
                           0,
                           0,
                           (__int64)&v34,
                           (__int64)&v28) )
      {
        if ( a5 )
        {
          v19 = v35;
          sub_6E4BC0((__int64)&v34, a5);
          v35 = v19 & 5 | v35 & 0xFA;
        }
        sub_7022F0(
          &v34,
          a1,
          v28,
          1,
          0,
          0,
          0,
          0,
          (__int64 *)&dword_4F077C8,
          (_DWORD *)a1 + 17,
          &dword_4F077C8,
          a6,
          0,
          &v29);
        goto LABEL_18;
      }
LABEL_17:
      sub_6E6260((_QWORD *)a6);
      goto LABEL_18;
    }
  }
  if ( (_BYTE)v17 != 2 || (v23 = *(_QWORD *)(v16 + 88)) == 0 || *(_BYTE *)(v23 + 173) != 12 )
  {
LABEL_28:
    if ( (unsigned int)sub_6E5430() )
      sub_686470(0xA75u, (_DWORD *)a1 + 17, (__int64)a2, *a1);
    goto LABEL_17;
  }
  v26 = v16;
  v28 = (_QWORD *)sub_6F6D20(v27, 0);
  sub_6E6A50(*(_QWORD *)(v26 + 88), (__int64)&v34);
  sub_82F1E0(a1, 0, &v34);
  if ( a5 )
  {
    v24 = v35;
    sub_6E4BC0((__int64)&v34, a5);
    v35 = v24 & 5 | v35 & 0xFA;
  }
  sub_7022F0(&v34, a1, v28, 1, 0, 0, 0, 0, (__int64 *)&dword_4F077C8, (_DWORD *)a1 + 17, &dword_4F077C8, a6, 0, &v29);
  if ( v29 && *(_BYTE *)(v29 + 56) == 105 )
    *(_BYTE *)(v29 + 56) = 106;
LABEL_18:
  if ( *(_BYTE *)(a6 + 16) )
  {
    v20 = *(_QWORD *)a6;
    for ( i = *(_BYTE *)(*(_QWORD *)a6 + 140LL); i == 12; i = *(_BYTE *)(v20 + 140) )
      v20 = *(_QWORD *)(v20 + 160);
    if ( i )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
      {
        sub_6E68E0(0x1Cu, a6);
      }
      else if ( (unsigned int)sub_6E91E0(0x1Cu, (_DWORD *)(a6 + 68)) )
      {
        sub_6E6840(a6);
      }
    }
  }
  return sub_6E26D0(2, a6);
}
