// Function: sub_702E30
// Address: 0x702e30
//
__int64 __fastcall sub_702E30(__m128i *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  _BYTE *v4; // rdx
  int v5; // r8d

  result = sub_6FB4D0((__int64)a1, a2);
  if ( (_DWORD)result )
  {
    v3 = sub_8D46C0(a1->m128i_i64[0]);
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v3) )
      sub_8AE000(v3);
    if ( (unsigned int)sub_8D25A0(v3) )
    {
      v5 = sub_8D2BE0(v3);
      result = 1;
      if ( v5 )
      {
        sub_6E6930(0xD56u, (__int64)a1, v3);
        return 0;
      }
    }
    else if ( dword_4F077BC
           && !(_DWORD)qword_4F077B4
           && qword_4F077A8
           && ((v4 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64), dword_4F04C44 != -1)
            || (v4[6] & 6) != 0
            || v4[4] == 12)
           && (v4[12] & 0x10) == 0 )
    {
      sub_684B30(0x354u, &a1[4].m128i_i32[1]);
      sub_6F4200(a1, dword_4D03B80, 0, 1);
      return 1;
    }
    else
    {
      sub_6E68E0(0x354u, (__int64)a1);
      return 0;
    }
  }
  return result;
}
