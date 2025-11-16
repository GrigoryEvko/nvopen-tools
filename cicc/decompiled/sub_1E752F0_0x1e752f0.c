// Function: sub_1E752F0
// Address: 0x1e752f0
//
__int64 *__fastcall sub_1E752F0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  _BYTE *v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdi
  char v7; // al
  __int64 v9; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-30h] BYREF
  char v11; // [rsp+20h] [rbp-20h]

  v9 = a2;
  v3 = *(_BYTE **)(a1 + 48);
  if ( v3 == *(_BYTE **)(a1 + 56) )
  {
    sub_1CFD630(a1 + 40, v3, &v9);
    v4 = *(_BYTE **)(a1 + 48);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = v9;
      v3 = *(_BYTE **)(a1 + 48);
    }
    v4 = v3 + 8;
    *(_QWORD *)(a1 + 48) = v4;
  }
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_BYTE *)(a1 + 32);
  v10[1] = *(_QWORD *)(a1 + 24);
  v10[0] = v5;
  v11 = v7;
  return sub_1E6D460(v6, ((__int64)&v4[-v6] >> 3) - 1, 0, *((_QWORD *)v4 - 1), (__int64)v10);
}
