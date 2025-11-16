// Function: sub_16F6490
// Address: 0x16f6490
//
bool __fastcall sub_16F6490(unsigned __int64 a1, __int64 a2)
{
  _BYTE *v2; // rsi
  _BYTE *v3; // rdi
  _BYTE *v4; // rax

  v2 = (_BYTE *)(a2 - 1);
  if ( (unsigned __int64)v2 < a1 )
    return 0;
  v3 = (_BYTE *)(a1 - 1);
  v4 = v2;
  do
  {
    if ( *v4 != 92 )
      break;
    --v4;
  }
  while ( v4 != v3 );
  return (v2 - v4) % 2 == 1;
}
