// Function: sub_67D610
// Address: 0x67d610
//
_DWORD *__fastcall sub_67D610(unsigned int a1, _DWORD *a2, unsigned __int8 a3)
{
  int v3; // eax
  _DWORD *v4; // rax
  _DWORD *v5; // r12
  unsigned __int8 v6; // al
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  __int64 v9; // rax
  unsigned __int8 v11; // [rsp+7h] [rbp-39h] BYREF
  int v12; // [rsp+8h] [rbp-38h] BYREF
  int v13; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  char v15[40]; // [rsp+18h] [rbp-28h] BYREF

  v3 = -1;
  if ( a3 != 6 && (unsigned __int8)(a3 - 9) > 2u )
    v3 = -(dword_4F07588 == 0);
  dword_4D03A00 = v3;
  v11 = a3;
  v4 = (_DWORD *)sub_67B9F0();
  *v4 = 0;
  v5 = v4;
  v4[44] = a1;
  v6 = v11;
  if ( v11 <= 7u )
  {
    sub_67C4B0((int *)a1, (char *)&v11, a2);
    v6 = v11;
  }
  *((_BYTE *)v5 + 180) = v6;
  *((_QWORD *)v5 + 11) = unk_4D03FF0;
  *((_QWORD *)v5 + 14) = sub_729E00((unsigned int)*a2, &v14, v15, &v12, &v13);
  *((_QWORD *)v5 + 15) = v14;
  v5[32] = v12;
  v7 = _mm_loadu_si128((const __m128i *)(v5 + 30));
  v5[26] = v13;
  v8 = _mm_loadu_si128((const __m128i *)(v5 + 26));
  *((_QWORD *)v5 + 12) = *(_QWORD *)a2;
  v9 = *(_QWORD *)a2;
  *((__m128i *)v5 + 9) = v8;
  *((_QWORD *)v5 + 17) = v9;
  *((__m128i *)v5 + 10) = v7;
  return v5;
}
