// Function: sub_6A9D40
// Address: 0x6a9d40
//
unsigned int *__fastcall sub_6A9D40(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  unsigned int *result; // rax

  if ( a1 )
  {
    v2 = *(unsigned __int8 *)(*a1 + 56LL);
  }
  else
  {
    switch ( word_4F06418[0] )
    {
      case 0xCFu:
        v2 = 13;
        break;
      case 0xD1u:
        v2 = 15;
        break;
      case 0xD2u:
        v2 = 111;
        break;
      case 0xD3u:
        v2 = 108;
        break;
      case 0x120u:
        v2 = 66;
        break;
      case 0x121u:
        v2 = 109;
        break;
      case 0x122u:
        v2 = 110;
        break;
      case 0x123u:
        v2 = 67;
        break;
      case 0x124u:
        v2 = 68;
        break;
      case 0x12Au:
        v2 = 72;
        break;
      case 0x12Bu:
        v2 = 73;
        break;
      case 0x12Cu:
        v2 = 74;
        break;
      default:
        sub_721090(0);
    }
  }
  v3 = sub_68AFD0(v2);
  sub_6A9320(a1, v2, v3, 1, 1u, 0, a2);
  result = &dword_4D044B0;
  if ( !dword_4D044B0 )
    return (unsigned int *)sub_6E6840(a2);
  return result;
}
