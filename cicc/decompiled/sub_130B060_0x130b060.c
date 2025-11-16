// Function: sub_130B060
// Address: 0x130b060
//
_BYTE *__fastcall sub_130B060(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  _BYTE *result; // rax

  v3 = *a2;
  result = (_BYTE *)sub_130AF40((__int64)a2);
  if ( (_BYTE)result )
  {
    sub_130ACF0("<jemalloc>: Error re-initializing mutex in child\n", v3);
    result = byte_4F969A5;
    if ( byte_4F969A5[0] )
      abort();
  }
  return result;
}
