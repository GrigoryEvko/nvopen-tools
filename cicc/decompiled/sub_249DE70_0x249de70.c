// Function: sub_249DE70
// Address: 0x249de70
//
__int64 __fastcall sub_249DE70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdi
  char v9; // al
  char v11; // [rsp+Fh] [rbp-31h]

  v3 = a1 + 32;
  v5 = a3 + 24;
  v6 = *(_QWORD *)(a3 + 32);
  v7 = a1 + 80;
  if ( v6 == v5 )
    goto LABEL_16;
  v11 = 0;
  do
  {
    v8 = v6 - 56;
    if ( !v6 )
      v8 = 0;
    if ( !sub_B2FC80(v8) && !(unsigned __int8)sub_B2D610(v8, 31) )
    {
      v9 = *(_BYTE *)(v8 + 32) & 0xF;
      if ( ((v9 + 14) & 0xFu) <= 3 || ((v9 + 7) & 0xFu) <= 1 )
      {
        if ( (unsigned __int8)sub_B2D610(v8, 3) )
          sub_B2D470(v8, 3);
        sub_B2CD30(v8, 31);
        v11 = 1;
      }
    }
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v5 != v6 );
  v3 = a1 + 32;
  v7 = a1 + 80;
  if ( !v11 )
  {
LABEL_16:
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v7;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v3;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v7;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  return a1;
}
