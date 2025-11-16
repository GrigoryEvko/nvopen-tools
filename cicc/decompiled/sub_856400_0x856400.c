// Function: sub_856400
// Address: 0x856400
//
_QWORD *__fastcall sub_856400(__int64 a1, _QWORD *a2, _QWORD *a3, int a4, char a5)
{
  char v6; // bl
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rdx
  __m128i *v11; // r12
  int v12; // r8d
  __int64 v13; // r8
  char v14; // al
  int v15; // edx
  __int64 m128i_i64; // r14
  int v18; // edx
  void (__fastcall *v19)(__m128i *); // rax
  int v20; // r15d

  v6 = 4 * (a5 & 1);
  v7 = sub_853DF0(a1);
  v10 = (__int64)a3;
  v11 = (__m128i *)v7;
  v12 = *(unsigned __int8 *)(v7 + 72);
  *(_QWORD *)(v7 + 48) = *a3;
  v13 = v12 & 0xFFFFFFF9;
  *(_QWORD *)(v7 + 56) = *a2;
  *(_BYTE *)(v7 + 72) = v13 | (2 * (a4 & 1)) | v6;
  v14 = *(_BYTE *)(a1 + 17);
  if ( v14 >= 0 )
  {
    if ( (*(_BYTE *)(a1 + 18) & 0x10) != 0 )
    {
      v18 = a4;
      m128i_i64 = (__int64)v11[1].m128i_i64;
      unk_4D03D00 = 1;
      sub_8555D0((__int64)v11, a1, v18);
      unk_4D03D00 = 0;
      if ( (*(_BYTE *)(a1 + 17) & 0x10) == 0 )
        goto LABEL_4;
    }
    else
    {
      v15 = a4;
      m128i_i64 = (__int64)v11[1].m128i_i64;
      sub_8555D0((__int64)v11, a1, v15);
      if ( (*(_BYTE *)(a1 + 17) & 0x10) == 0 )
      {
LABEL_4:
        sub_7AE110(v11[1].m128i_i64[1], (_QWORD **)&v11[1].m128i_i64[1], m128i_i64);
        goto LABEL_5;
      }
    }
    sub_7C9CD0(v11[3].m128i_i32, 0);
    sub_7C9730(m128i_i64);
    v11[5].m128i_i64[0] = (__int64)sub_7C9CF0();
    goto LABEL_4;
  }
  if ( *(_DWORD *)(a1 + 12) == 5 )
  {
    v10 = v14 & 0x18;
    if ( (_BYTE)v10 != 24 )
      goto LABEL_11;
  }
  v20 = dword_4D03D1C;
  dword_4D03D1C = (v14 & 0x20) != 0;
  sub_856220(a4, (__int64)a2, v10, v8, v13, v9);
  dword_4D03D1C = v20;
  v11[5].m128i_i64[0] = (__int64)sub_724840(dword_4F073B8[0], (const char *)qword_4F5FC08);
LABEL_5:
  if ( *(_DWORD *)(a1 + 12) != 5 )
    return sub_854040((__int64)v11);
LABEL_11:
  v19 = (void (__fastcall *)(__m128i *))off_4A51EC0[*(unsigned __int8 *)(a1 + 16)];
  if ( v19 )
    v19(v11);
  if ( (*(_BYTE *)(a1 + 17) & 8) != 0 )
  {
    v11[4].m128i_i64[0] = sub_869D30();
    sub_8540F0(v11, 0, 0, 1);
  }
  return sub_853F90(v11);
}
