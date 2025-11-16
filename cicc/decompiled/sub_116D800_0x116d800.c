// Function: sub_116D800
// Address: 0x116d800
//
__int64 __fastcall sub_116D800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 *v13; // rdx
  __int64 *v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  _QWORD v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(_QWORD *)(**(_QWORD **)(a3 - 8) + 48LL);
  v19[0] = v5;
  if ( v5 )
  {
    v6 = a2 + 48;
    sub_B96E90((__int64)v19, v5, 1);
    v7 = *(_QWORD *)(a2 + 48);
    if ( !v7 )
      goto LABEL_4;
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 48);
    v6 = a2 + 48;
    if ( !v7 )
      goto LABEL_6;
  }
  sub_B91220(v6, v7);
LABEL_4:
  v8 = (unsigned __int8 *)v19[0];
  *(_QWORD *)(a2 + 48) = v19[0];
  if ( v8 )
    sub_B976B0((__int64)v19, v8, v6);
LABEL_6:
  v9 = 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v10 = *(_QWORD *)(a3 - 8);
    v11 = v10 + v9;
  }
  else
  {
    v11 = a3;
    v10 = a3 - v9;
  }
  result = sub_116D080(v10, v11, 1);
  v14 = v13;
  v15 = (__int64 *)result;
  if ( (__int64 *)result != v13 )
  {
    do
    {
      v16 = *v15;
      v15 += 4;
      v17 = sub_B10CD0(v16 + 48);
      v18 = sub_B10CD0(v6);
      result = sub_AE8F10(a2, v18, v17);
    }
    while ( v14 != v15 );
  }
  return result;
}
