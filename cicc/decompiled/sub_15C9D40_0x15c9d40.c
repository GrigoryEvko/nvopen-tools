// Function: sub_15C9D40
// Address: 0x15c9d40
//
void __fastcall sub_15C9D40(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4)
{
  char *v5; // rsi
  unsigned __int64 v6; // rax
  char v7; // [rsp+14h] [rbp-1Ch] BYREF
  _BYTE v8[27]; // [rsp+15h] [rbp-1Bh] BYREF

  *(_QWORD *)a1 = a1 + 16;
  if ( a2 )
  {
    sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  if ( a4 )
  {
    v5 = v8;
    do
    {
      *--v5 = a4 % 0xA + 48;
      v6 = a4;
      a4 /= 0xAu;
    }
    while ( v6 > 9 );
  }
  else
  {
    v7 = 48;
    v5 = &v7;
  }
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_15C7F50((__int64 *)(a1 + 32), v5, (__int64)v8);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
}
