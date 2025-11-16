// Function: sub_916690
// Address: 0x916690
//
__int64 __fastcall sub_916690(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rsi
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 result; // rax
  _BYTE v14[33]; // [rsp+Fh] [rbp-21h] BYREF

  v5 = sub_90A440((__int64)a1, a2, v14);
  if ( (*(_BYTE *)(a2 + 156) & 2) != 0 )
  {
    v10 = sub_91DAF0(a1, *(_QWORD *)(a2 + 120));
LABEL_15:
    if ( a3 )
      goto LABEL_16;
    goto LABEL_12;
  }
  v9 = v5;
  if ( v14[0] == 2 )
  {
    if ( v5 )
      goto LABEL_11;
    goto LABEL_14;
  }
  if ( !v14[0] || v14[0] == 3 )
  {
LABEL_14:
    v10 = sub_91DAD0(a1, *(_QWORD *)(a2 + 120));
    goto LABEL_15;
  }
  if ( !v5 )
    goto LABEL_8;
  if ( v14[0] != 1 )
  {
    v9 = v5 + 64;
LABEL_8:
    sub_91B980("unsupported initialization variant!", v9);
  }
LABEL_11:
  v10 = sub_91FFA0(a1, v5, *(_QWORD *)(a2 + 120), v6, v7, v8);
  if ( a3 )
    goto LABEL_16;
LABEL_12:
  if ( v10 )
    a3 = *(_QWORD *)(v10 + 8);
  a3 = sub_916620(a1, a2, a3);
LABEL_16:
  sub_90A710((__int64)a1, a3, v10, a2);
  LODWORD(v11) = sub_91CB50(a2);
  v12 = 0;
  if ( (_DWORD)v11 )
  {
    _BitScanReverse64((unsigned __int64 *)&v11, (unsigned int)v11);
    LOBYTE(v12) = 63 - (v11 ^ 0x3F);
    BYTE1(v12) = 1;
  }
  sub_B2F740(a3, v12);
  result = sub_91B5A0(a2);
  if ( (_BYTE)result )
  {
    if ( (*(_BYTE *)(a2 + 156) & 2) == 0 )
      return sub_914140((__int64)a1, a3);
  }
  return result;
}
