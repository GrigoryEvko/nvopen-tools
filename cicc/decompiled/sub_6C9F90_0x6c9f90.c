// Function: sub_6C9F90
// Address: 0x6c9f90
//
__int64 __fastcall sub_6C9F90(__int64 a1, unsigned int a2, unsigned int a3, __m128i *a4, __m128i *a5, _DWORD *a6)
{
  char v9; // bl
  char v10; // al
  __int64 v11; // r8
  bool v12; // bl
  __int64 v13; // rdx
  __m128i *v14; // rsi
  __int64 v15; // rcx
  __int64 result; // rax
  int v17; // eax
  __m128i *v19; // [rsp+10h] [rbp-40h]
  unsigned int v21; // [rsp+18h] [rbp-38h]

  v9 = *(_BYTE *)(qword_4D03C50 + 21LL);
  v10 = sub_8D2600(a1);
  v11 = qword_4D03C50;
  v12 = (v9 & 0x20) != 0;
  *(_BYTE *)(qword_4D03C50 + 21LL) = (32 * (v10 & 1)) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xDF;
  v13 = a3;
  v14 = a5;
  if ( !dword_4D04964
    && (*(_BYTE *)(v11 + 19) & 0x40) != 0
    && (unsigned __int8)(*(_BYTE *)(v11 + 16) - 1) <= 1u
    && (v19 = a5, v21 = v13, v17 = sub_8D2780(a1), v13 = v21, v14 = v19, v17) )
  {
    sub_6BA150(a2 ^ 1, a2, 1, 0, v21, (__int64)a4, 0, a6);
    if ( a2 && !*a6 )
LABEL_12:
      sub_6E45A0(a4);
  }
  else
  {
    if ( a2 )
    {
      sub_6B9CA0(2, 1, a4, v14, a6);
      if ( *a6 )
        goto LABEL_7;
      goto LABEL_12;
    }
    v15 = 3;
    if ( dword_4F077C4 == 2 )
      v15 = 19;
    sub_69ED20((__int64)a4, v14, v13, v15);
  }
LABEL_7:
  result = *(_BYTE *)(qword_4D03C50 + 21LL) & 0xDF;
  *(_BYTE *)(qword_4D03C50 + 21LL) = *(_BYTE *)(qword_4D03C50 + 21LL) & 0xDF | (32 * v12);
  return result;
}
