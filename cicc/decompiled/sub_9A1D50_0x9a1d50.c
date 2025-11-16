// Function: sub_9A1D50
// Address: 0x9a1d50
//
__int16 __fastcall sub_9A1D50(unsigned __int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, int a5)
{
  unsigned __int8 v8; // dl
  __int64 v9; // rdi
  __int16 result; // ax

  v9 = sub_984CA0(a4);
  result = 0;
  if ( v9 )
    return sub_9A13D0(v9, a1, a2, a3, a5, v8, 0);
  return result;
}
