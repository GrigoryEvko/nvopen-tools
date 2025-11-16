// Function: sub_2FF7F40
// Address: 0x2ff7f40
//
__int64 __fastcall sub_2FF7F40(__int64 a1, __int64 a2, _WORD *a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx

  if ( sub_2FF7B90(a1) )
  {
    v4 = *(_QWORD *)(a1 + 184);
    if ( v4 )
    {
      result = (unsigned int)*(__int16 *)(v4 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 6LL));
      if ( (int)result < 0 )
        return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 200) + 1104LL))(
                 *(_QWORD *)(a1 + 200),
                 a1 + 80,
                 a2);
    }
    else
    {
      return 1;
    }
  }
  else
  {
    if ( !sub_2FF7B70(a1) )
      goto LABEL_10;
    if ( !a3 )
      a3 = sub_2FF7DB0(a1, a2);
    LOWORD(result) = *a3 & 0x1FFF;
    if ( (_WORD)result == 0x1FFF )
    {
LABEL_10:
      v6 = *(unsigned __int16 *)(a2 + 68);
      if ( !(_WORD)v6 )
        return 0;
      v7 = (unsigned int)(v6 - 9);
      if ( (unsigned __int16)v7 <= 0x3Bu && (v8 = 0x800000000000C09LL, _bittest64(&v8, v7)) )
        return 0;
      else
        return (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x10LL) == 0;
    }
    else
    {
      return (unsigned __int16)result;
    }
  }
  return result;
}
