// Function: sub_15C9B00
// Address: 0x15c9b00
//
void __fastcall sub_15C9B00(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rdi
  bool v7; // zf
  char *v8; // r9
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  _BYTE *v12; // rsi
  _BYTE *v13; // r10
  unsigned __int64 v14; // rax
  char v15; // [rsp+14h] [rbp-1Ch] BYREF
  _BYTE v16[27]; // [rsp+15h] [rbp-1Bh] BYREF

  *(_QWORD *)a1 = a1 + 16;
  if ( !a2 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    v6 = (__int64 *)(a1 + 32);
    v7 = a4 == 0;
    if ( a4 >= 0 )
      goto LABEL_3;
LABEL_11:
    v11 = -a4;
    v12 = v16;
    do
    {
      v13 = v12--;
      *v12 = v11 % 0xA + 48;
      v14 = v11;
      v11 /= 0xAu;
    }
    while ( v14 > 9 );
    *(v12 - 1) = 45;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    sub_15C7F50(v6, v13 - 2, (__int64)v16);
    goto LABEL_6;
  }
  sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  v6 = (__int64 *)(a1 + 32);
  v7 = a4 == 0;
  if ( a4 < 0 )
    goto LABEL_11;
LABEL_3:
  if ( v7 )
  {
    v15 = 48;
    v8 = &v15;
  }
  else
  {
    v9 = a4;
    v8 = v16;
    do
    {
      *--v8 = v9 % 0xA + 48;
      v10 = v9;
      v9 /= 0xAu;
    }
    while ( v10 > 9 );
  }
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_15C7F50(v6, v8, (__int64)v16);
LABEL_6:
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
}
