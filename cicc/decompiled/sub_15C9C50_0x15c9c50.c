// Function: sub_15C9C50
// Address: 0x15c9c50
//
void __fastcall sub_15C9C50(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v5; // rcx
  char *v6; // r8
  unsigned __int64 v7; // rax
  char v8; // [rsp+14h] [rbp-1Ch] BYREF
  _BYTE v9[27]; // [rsp+15h] [rbp-1Bh] BYREF

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
  v5 = a4;
  if ( a4 )
  {
    v6 = v9;
    do
    {
      *--v6 = v5 % 0xA + 48;
      v7 = v5;
      v5 /= 0xAu;
    }
    while ( v7 > 9 );
  }
  else
  {
    v8 = 48;
    v6 = &v8;
  }
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_15C7F50((__int64 *)(a1 + 32), v6, (__int64)v9);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
}
