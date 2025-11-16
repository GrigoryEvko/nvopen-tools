// Function: sub_6E5270
// Address: 0x6e5270
//
__int64 __fastcall sub_6E5270(__int64 a1, __int64 a2, _DWORD *a3, __m128i *a4)
{
  _DWORD *v4; // r13
  __int64 result; // rax

  if ( !qword_4D03C50 )
  {
    if ( !qword_4F04C50 || (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 4) == 0 )
    {
      if ( (*(_BYTE *)(a1 + 206) & 0x10) == 0 )
      {
LABEL_8:
        v4 = sub_67E020(0xB75u, a3, *(_QWORD *)a1);
        sub_67E370((__int64)v4, a4);
        sub_685910((__int64)v4, (FILE *)a4);
      }
LABEL_9:
      result = 1;
      if ( a2 )
      {
        sub_72C970(a2);
        return 1;
      }
      return result;
    }
LABEL_13:
    sub_7604D0(a1, 11);
    return 0;
  }
  if ( (*(_QWORD *)(qword_4D03C50 + 16LL) & 0x200004000LL) != 0
    || qword_4F04C50 && (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 4) != 0 )
  {
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 21LL) & 4) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 206) & 0x10) != 0 )
      goto LABEL_9;
    if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
      goto LABEL_8;
    sub_6E50A0();
    goto LABEL_9;
  }
  if ( unk_4D03C20 )
  {
    sub_67E3D0(a4);
  }
  else
  {
    unk_4D03C20 = a1;
    unk_4D03C28 = *(_QWORD *)a3;
    unk_4D03C30 = _mm_loadu_si128(a4);
    a4->m128i_i64[0] = 0;
    a4->m128i_i64[1] = 0;
  }
  return 0;
}
