// Function: sub_130FD00
// Address: 0x130fd00
//
__int64 __fastcall sub_130FD00(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned int v6; // r13d
  __int64 v7; // r11
  unsigned int v8; // r15d
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  *a2 = a1;
  *(_OWORD *)a1 = 0;
  v4 = unk_5060A18;
  v5 = 36;
  *(_QWORD *)(a1 + 168) = a2;
  *(_QWORD *)(a1 + 160) = a3;
  *(_DWORD *)(a1 + 48) = 0;
  if ( (unsigned int)v4 >= 0x24 )
    v5 = v4;
  *(_QWORD *)(a1 + 40) = 0;
  memset(a2 + 1, 0, 24 * v5);
  v14 = 0;
  sub_131D3D0(unk_5060A20, (unsigned int)v4, a3, &v14);
  v6 = unk_5060A18;
  if ( !unk_5060A18 )
    goto LABEL_16;
  v7 = a1;
  v8 = 0;
  do
  {
    v9 = v8;
    if ( v8 <= 0x23 )
    {
      *(_BYTE *)(v7 + v8 + 52) = 1;
      *(_BYTE *)(v7 + v8 + 88) = 0;
      v10 = qword_4F96AD0 / qword_505FA40[v8];
      if ( v10 > 0xFF )
        LOBYTE(v10) = -1;
      *(_BYTE *)(v7 + v8 + 124) = v10;
    }
    v13 = v7;
    ++v8;
    sub_131D410(&a2[3 * v9 + 1], 2 * v9 + unk_5060A20, a3, &v14);
    v7 = v13;
    v6 = unk_5060A18;
  }
  while ( v8 < unk_5060A18 );
  if ( unk_5060A18 <= 0x23u )
  {
LABEL_16:
    do
    {
      v11 = v6++;
      v15[0] = 0;
      sub_131D410(&a2[3 * v11 + 1], 2 * v11 + unk_5060A20, a3, v15);
    }
    while ( v6 != 36 );
    v6 = unk_5060A18;
  }
  return sub_131D3F0(unk_5060A20, v6, a3, &v14);
}
