// Function: sub_702F90
// Address: 0x702f90
//
__int64 __fastcall sub_702F90(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char v7; // cl
  __int64 result; // rax
  __int64 v9; // rax
  char i; // dl
  int v11; // r8d
  _BOOL4 v12; // eax
  __int64 v13; // rax
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( HIDWORD(qword_4F077B4) )
    sub_6FA330(a1, 1u);
  if ( a1[1].m128i_i8[0] == 2 && a1[19].m128i_i8[13] == 12 && a1[20].m128i_i8[0] == 1 )
    sub_6F3DD0((__int64)a1, 1, 0, a4, a5, a6);
  v6 = a1->m128i_i64[0];
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v6) )
    sub_8AE000(v6);
  if ( a1[1].m128i_i8[1] == 1 && !sub_6ED0A0((__int64)a1) && !(unsigned int)sub_8D23B0(v6) )
  {
    if ( (unsigned int)sub_6E9790((__int64)a1, v14) )
    {
      v12 = (*(_BYTE *)(v14[0] + 156LL) & 4) != 0;
    }
    else
    {
      if ( a1[1].m128i_i8[0] != 1 )
        goto LABEL_19;
      v12 = sub_6E9180(a1[9].m128i_i64[0]);
    }
    if ( v12 )
      goto LABEL_27;
LABEL_19:
    if ( (*(_BYTE *)(v6 + 140) & 0xFB) != 8 || (sub_8D4C10(v6, dword_4F077C4 != 2) & 1) == 0 )
      goto LABEL_20;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 <= 0x222DFu )
        {
LABEL_42:
          if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
            goto LABEL_27;
LABEL_20:
          if ( unk_4D0436C )
          {
            if ( a1[1].m128i_i8[0] == 1 )
            {
              v13 = a1[9].m128i_i64[0];
              if ( *(_BYTE *)(v13 + 24) == 1 && *(_BYTE *)(v13 + 56) == 6 )
                sub_69D070(0x89u, &a1[4].m128i_i32[1]);
            }
          }
          v11 = sub_8D3A70(v6);
          result = 1;
          if ( !v11 )
            return result;
          while ( *(_BYTE *)(v6 + 140) == 12 )
            v6 = *(_QWORD *)(v6 + 160);
          if ( (*(_BYTE *)(v6 + 176) & 2) == 0 )
            return 1;
        }
LABEL_27:
        v7 = 1;
        goto LABEL_7;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_27;
    }
    if ( qword_4F077A0 <= 0x1869Fu )
      goto LABEL_42;
    goto LABEL_27;
  }
  v7 = 0;
LABEL_7:
  result = 0;
  if ( a1[1].m128i_i8[0] )
  {
    v9 = a1->m128i_i64[0];
    for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v9 + 140) )
      v9 = *(_QWORD *)(v9 + 160);
    result = 0;
    if ( i )
    {
      if ( !dword_4F077C0 || !v7 )
      {
        sub_6E68E0(0x89u, (__int64)a1);
        return 0;
      }
      sub_6E5C80(7, 0x89u, &a1[4].m128i_i32[1]);
      return 1;
    }
  }
  return result;
}
