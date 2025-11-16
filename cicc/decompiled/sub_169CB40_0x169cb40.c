// Function: sub_169CB40
// Address: 0x169cb40
//
void *__fastcall sub_169CB40(__int64 a1, char a2, char a3, __int64 *a4, float a5)
{
  _BYTE *v8; // rdi

  v8 = (_BYTE *)(a1 + 8);
  if ( *(void **)(a1 + 8) == sub_16982C0() )
    return (void *)sub_169CAA0((__int64)v8, a2, a3, a4, a5);
  else
    return sub_16986F0(v8, a2, a3, a4);
}
