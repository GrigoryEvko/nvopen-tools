// Function: sub_FFE7D0
// Address: 0xffe7d0
//
__int64 __fastcall sub_FFE7D0(int a1, __int64 a2, unsigned __int8 *a3, int a4, __int64 a5)
{
  unsigned __int16 v7; // ax
  __int64 v8; // rdi
  __int64 result; // rax

  v7 = sub_9A1D50(0x20u, a2, a3, a5, a4);
  if ( !HIBYTE(v7) || !(_BYTE)v7 )
    return 0;
  v8 = *(_QWORD *)(a2 + 8);
  switch ( a1 )
  {
    case 15:
    case 22:
    case 23:
    case 30:
      result = sub_AD6530(v8, a2);
      break;
    case 19:
    case 20:
      result = sub_AD64C0(v8, 1, 0);
      break;
    case 28:
    case 29:
      result = (__int64)a3;
      break;
    default:
      return 0;
  }
  return result;
}
