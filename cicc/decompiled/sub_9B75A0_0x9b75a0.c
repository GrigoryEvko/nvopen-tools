// Function: sub_9B75A0
// Address: 0x9b75a0
//
char __fastcall sub_9B75A0(__int64 a1, unsigned int a2, __int64 a3)
{
  char result; // al
  unsigned __int64 v5; // r12
  __int64 v6; // rax

  if ( a3 && (unsigned __int8)sub_B60C40(a1) )
    return sub_DFAA40(a3, (unsigned int)a1, a2);
  if ( (_DWORD)a1 == 285 )
    return a2 == 1;
  if ( (unsigned int)a1 <= 0x11D )
  {
    if ( (_DWORD)a1 == 166 )
      return a2 == 2 || a2 - 4 <= 1;
    if ( (unsigned int)a1 <= 0xA6 )
    {
      if ( (_DWORD)a1 == 67 || (a1 & 0xFFFFFFBF) == 1 )
        return a2 == 1;
      return 0;
    }
    if ( (_DWORD)a1 != 207 )
      return 0;
    return a2 == 1;
  }
  if ( (unsigned int)a1 > 0x170 )
  {
    v5 = (unsigned int)(a1 - 404);
    if ( (unsigned int)v5 <= 0x20 )
    {
      v6 = 0x100000501LL;
      if ( _bittest64(&v6, v5) )
        return a2 == 1;
    }
    return 0;
  }
  if ( (unsigned int)a1 > 0x16E )
    return a2 == 2;
  result = 0;
  if ( (unsigned int)(a1 - 331) <= 1 )
    return a2 == 2;
  return result;
}
