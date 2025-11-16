// Function: sub_1196B10
// Address: 0x1196b10
//
bool __fastcall sub_1196B10(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // rdx
  bool result; // al
  __int64 v4; // r8
  int v5; // edx
  __int64 v6; // rax

  v2 = *a2;
  result = 0;
  if ( (unsigned __int8)v2 <= 0x1Cu )
  {
    if ( (_BYTE)v2 != 5 )
      return result;
    v5 = *((unsigned __int16 *)a2 + 1);
    result = (*((_WORD *)a2 + 1) & 0xFFFD) == 13 || (v5 & 0xFFF7) == 17;
    if ( !result )
      return result;
  }
  else
  {
    if ( (unsigned __int8)v2 > 0x36u )
      return result;
    v4 = 0x40540000000000LL;
    if ( !_bittest64(&v4, v2) )
      return result;
    v5 = (unsigned __int8)v2 - 29;
  }
  result = 0;
  if ( v5 == 25 )
  {
    result = (a2[1] & 2) != 0;
    if ( (a2[1] & 2) != 0 )
    {
      v6 = *((_QWORD *)a2 - 8);
      if ( v6 )
      {
        **(_QWORD **)a1 = v6;
        return *((_QWORD *)a2 - 4) == *(_QWORD *)(a1 + 8);
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
