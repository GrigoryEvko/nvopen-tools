// Function: sub_648C10
// Address: 0x648c10
//
__int64 __fastcall sub_648C10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  char v5; // al

  result = (__int64)&dword_4D04338;
  if ( dword_4D04338 )
    return result;
  if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || !qword_4F077A8 )
  {
    v4 = 799;
    if ( dword_4D04964 )
      return sub_6853B0(7, v4, a2, a1);
    goto LABEL_5;
  }
  v5 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v5 - 4) <= 1u )
  {
    result = word_4F06418[0];
    if ( word_4F06418[0] != 55 && word_4F06418[0] != 73 )
      return result;
    goto LABEL_5;
  }
  if ( v5 != 3 )
  {
LABEL_5:
    v4 = 795;
    return sub_6853B0(7, v4, a2, a1);
  }
  if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(a1 + 88)) )
    return sub_6853B0(7, 795, a2, a1);
  result = word_4F06418[0];
  if ( word_4F06418[0] == 55 || word_4F06418[0] == 73 )
    return sub_6853B0(7, 795, a2, a1);
  return result;
}
