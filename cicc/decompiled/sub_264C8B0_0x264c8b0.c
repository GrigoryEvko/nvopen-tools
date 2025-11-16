// Function: sub_264C8B0
// Address: 0x264c8b0
//
__int64 *__fastcall sub_264C8B0(__int64 a1)
{
  _QWORD **v1; // r12
  _QWORD **i; // rbx
  __int64 *result; // rax

  v1 = *(_QWORD ***)(a1 + 328);
  for ( i = *(_QWORD ***)(a1 + 320); v1 != i; ++i )
  {
    if ( *((_BYTE *)*i + 2) )
      result = sub_264C780(*i);
  }
  return result;
}
