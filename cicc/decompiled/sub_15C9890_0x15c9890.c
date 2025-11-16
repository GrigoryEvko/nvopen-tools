// Function: sub_15C9890
// Address: 0x15c9890
//
void __fastcall sub_15C9890(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4)
{
  int v4; // r12d
  __int64 *v6; // rdi
  char *v7; // r9
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  _BYTE *v10; // rsi
  _BYTE *v11; // r10
  unsigned __int64 v12; // rax
  char v13; // [rsp+14h] [rbp-1Ch] BYREF
  _BYTE v14[27]; // [rsp+15h] [rbp-1Bh] BYREF

  v4 = a4;
  *(_QWORD *)a1 = a1 + 16;
  if ( !a2 )
  {
    a4 = (int)a4;
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    v6 = (__int64 *)(a1 + 32);
    if ( (a4 & 0x80000000) == 0LL )
      goto LABEL_3;
LABEL_11:
    v9 = -(__int64)a4;
    v10 = v14;
    do
    {
      v11 = v10--;
      *v10 = v9 % 0xA + 48;
      v12 = v9;
      v9 /= 0xAu;
    }
    while ( v12 > 9 );
    *(v10 - 1) = 45;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    sub_15C7F50(v6, v11 - 2, (__int64)v14);
    goto LABEL_6;
  }
  sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  a4 = v4;
  v6 = (__int64 *)(a1 + 32);
  if ( v4 < 0LL )
    goto LABEL_11;
LABEL_3:
  if ( v4 )
  {
    v7 = v14;
    do
    {
      *--v7 = a4 % 0xA + 48;
      v8 = a4;
      a4 /= 0xAu;
    }
    while ( v8 > 9 );
  }
  else
  {
    v13 = 48;
    v7 = &v13;
  }
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_15C7F50(v6, v7, (__int64)v14);
LABEL_6:
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
}
