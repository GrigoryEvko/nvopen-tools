// Function: sub_10A4400
// Address: 0x10a4400
//
bool __fastcall sub_10A4400(__int64 a1, _BYTE *a2)
{
  bool result; // al
  unsigned __int8 *v3; // rcx
  unsigned __int64 v4; // rdx
  __int64 v5; // r8
  int v6; // edx
  __int64 v7; // rax

  result = 0;
  if ( *a2 == 68 )
  {
    v3 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v4 = *v3;
    if ( (unsigned __int8)v4 <= 0x1Cu )
    {
      if ( (_BYTE)v4 != 5 )
        return result;
      v6 = *((unsigned __int16 *)v3 + 1);
      result = (*((_WORD *)v3 + 1) & 0xFFF7) == 17 || (v6 & 0xFFFD) == 13;
      if ( !result )
        return result;
    }
    else
    {
      if ( (unsigned __int8)v4 > 0x36u )
        return result;
      v5 = 0x40540000000000LL;
      if ( !_bittest64(&v5, v4) )
        return result;
      v6 = (unsigned __int8)v4 - 29;
    }
    result = 0;
    if ( v6 == 15 )
    {
      result = (v3[1] & 2) != 0;
      if ( (v3[1] & 2) != 0 )
      {
        v7 = *((_QWORD *)v3 - 8);
        if ( v7 )
        {
          **(_QWORD **)a1 = v7;
          return *((_QWORD *)v3 - 4) == *(_QWORD *)(a1 + 8);
        }
        else
        {
          return 0;
        }
      }
    }
  }
  return result;
}
