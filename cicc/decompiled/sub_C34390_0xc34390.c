// Function: sub_C34390
// Address: 0xc34390
//
char __fastcall sub_C34390(__int64 a1, char a2, int a3, unsigned int a4)
{
  char result; // al
  __int64 v5; // rax

  switch ( a2 )
  {
    case 0:
      result = 0;
      break;
    case 1:
      result = 1;
      if ( a3 != 3 )
      {
        result = 0;
        if ( a3 == 2 && (*(_BYTE *)(a1 + 20) & 7) != 3 )
        {
          v5 = sub_C33930(a1);
          result = (unsigned int)sub_C45D90(v5, a4) != 0;
        }
      }
      break;
    case 2:
      result = ((*(_BYTE *)(a1 + 20) >> 3) ^ 1) & 1;
      break;
    case 3:
      result = (*(_BYTE *)(a1 + 20) & 8) != 0;
      break;
    case 4:
      result = (unsigned int)(a3 - 2) <= 1;
      break;
    default:
      BUG();
  }
  return result;
}
