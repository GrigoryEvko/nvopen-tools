// Function: sub_ADAB70
// Address: 0xadab70
//
__int64 __fastcall sub_ADAB70(int a1, unsigned __int64 a2, __int64 **a3, char a4)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case '&':
      result = sub_AD4C30(a2, a3, a4);
      break;
    case '/':
      result = sub_AD4C50(a2, a3, a4);
      break;
    case '0':
      result = sub_AD4C70(a2, a3, a4);
      break;
    case '1':
      result = sub_AD4C90(a2, a3, a4);
      break;
    case '2':
      result = sub_ADA8A0(a2, (__int64)a3, a4);
      break;
    default:
      BUG();
  }
  return result;
}
