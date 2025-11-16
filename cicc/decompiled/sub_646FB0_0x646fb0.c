// Function: sub_646FB0
// Address: 0x646fb0
//
__int64 __fastcall sub_646FB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v4; // rbx
  __int64 i; // r14
  __int64 v6; // rdi

  result = unk_4F04C50;
  v4 = *(_BYTE **)(unk_4F04C50 + 32LL);
  if ( dword_4F077C4 == 2 )
  {
    result = (__int64)&unk_4F07778;
    if ( unk_4F07778 > 202301 )
      return result;
  }
  if ( *(_BYTE *)(a1 + 136) <= 2u )
  {
    if ( (v4[193] & 5) != 0 )
      result = sub_6851C0(2661, a2);
LABEL_5:
    v4[193] &= ~2u;
    return result;
  }
  result = (__int64)&dword_4F04C44;
  if ( dword_4F04C44 != -1 )
    return result;
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(result + 6) & 6) != 0 || *(_BYTE *)(result + 4) == 12 )
    return result;
  for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D4160(i) )
  {
    result = (__int64)&dword_4F077BC;
    if ( dword_4F077BC )
    {
      result = (__int64)&qword_4F077B4;
      if ( !(_DWORD)qword_4F077B4 )
      {
        result = (__int64)&qword_4F077A8;
        if ( qword_4F077A8 )
        {
          result = (unsigned int)*(unsigned __int8 *)(i + 140) - 9;
          if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u && (*(_BYTE *)(i + 176) & 0x10) != 0 )
          {
            LOBYTE(result) = v4[195];
            goto LABEL_19;
          }
        }
      }
    }
  }
  else
  {
    result = (unsigned __int8)v4[195];
    if ( !(_DWORD)qword_4F077B4 || !qword_4F077A0 || (result & 1) == 0 && (v4[208] & 2) == 0 )
    {
LABEL_19:
      result &= 3u;
      if ( (_BYTE)result != 1 && (v4[193] & 5) != 0 )
      {
        v6 = (unsigned int)sub_8D4160(i) == 0 ? 2660 : 3137;
        sub_685360(v6, a2);
        result = sub_72C930(v6);
        *(_QWORD *)(a1 + 120) = result;
      }
      goto LABEL_5;
    }
  }
  if ( !*(_BYTE *)(a1 + 177) )
  {
    if ( dword_4F077C4 != 2 || (result = (__int64)&unk_4F07778, unk_4F07778 <= 202001) )
    {
      if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 2u
        || (*(_BYTE *)(i + 179) & 1) == 0
        || (result = (__int64)&dword_4D04964, dword_4D04964) )
      {
        if ( !(_DWORD)qword_4F077B4 || qword_4F077A0 <= 0x765Bu || (result = sub_729F80(dword_4F063F8), !(_DWORD)result) )
        {
          result = v4[195] & 3;
          if ( (_BYTE)result != 1 && (v4[193] & 5) != 0 )
            result = sub_6851C0(2662, a2);
          goto LABEL_5;
        }
      }
    }
  }
  return result;
}
