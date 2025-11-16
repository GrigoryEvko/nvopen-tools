// Function: sub_22329C0
// Address: 0x22329c0
//
__int64 __fastcall sub_22329C0(_BYTE *a1, unsigned __int64 a2, __int64 a3, __int16 a4, char a5)
{
  _BYTE *v5; // r9
  unsigned __int64 v6; // rcx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax

  v5 = a1;
  if ( a5 )
  {
    do
    {
      *--a1 = *(_BYTE *)(a3 + a2 % 0xA + 4);
      v6 = a2;
      a2 /= 0xAu;
    }
    while ( v6 > 9 );
  }
  else if ( (a4 & 0x4A) == 0x40 )
  {
    do
    {
      --a1;
      v10 = a2 & 7;
      a2 >>= 3;
      *a1 = *(_BYTE *)(a3 + v10 + 4);
    }
    while ( a2 );
  }
  else
  {
    v8 = a3 + (-(__int64)((a4 & 0x4000) == 0) & 0xFFFFFFFFFFFFFFF0LL) + 20;
    do
    {
      --a1;
      v9 = a2 & 0xF;
      a2 >>= 4;
      *a1 = *(_BYTE *)(v8 + v9);
    }
    while ( a2 );
  }
  return v5 - a1;
}
