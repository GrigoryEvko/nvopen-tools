// Function: sub_40D550
// Address: 0x40d550
//
_WORD *__fastcall sub_40D550(unsigned __int64 a1, char a2, char a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // rdx
  const char *v8; // r10
  __int64 v9; // r11
  _WORD *result; // rax
  __int64 v11; // rdx
  char *v12; // rdx

  *(_BYTE *)(a4 + 64) = 0;
  v7 = a4;
  v8 = "0123456789ABCDEF";
  LODWORD(a4) = 64;
  if ( !a3 )
    v8 = "0123456789abcdef";
  do
  {
    v9 = a1 & 0xF;
    a4 = (unsigned int)(a4 - 1);
    result = (_WORD *)(v7 + a4);
    a1 >>= 4;
    *(_BYTE *)(v7 + a4) = v8[v9];
  }
  while ( a1 );
  v11 = (unsigned int)(64 - a4);
  if ( a2 )
  {
    *a5 = v11 + 2;
    v12 = "0X";
    if ( !a3 )
      v12 = "0x";
    *--result = *(_WORD *)v12;
  }
  else
  {
    *a5 = v11;
  }
  return result;
}
