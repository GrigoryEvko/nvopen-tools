// Function: sub_14A8040
// Address: 0x14a8040
//
__int64 __fastcall sub_14A8040(__int64 a1, __int64 a2, __int64 *a3)
{
  _BYTE *v6; // rsi
  _BYTE *v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // r14
  unsigned int v10; // r12d
  bool v12[49]; // [rsp+Fh] [rbp-31h] BYREF

  v6 = *(_BYTE **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
  if ( v6 && (unsigned __int8)(*v6 - 4) >= 0x1Fu )
    v6 = 0;
  v7 = *(_BYTE **)(a1 + 8 * (1LL - *(unsigned int *)(a1 + 8)));
  if ( v7 && (unsigned __int8)(*v7 - 4) >= 0x1Fu )
    v7 = 0;
  v8 = sub_14A7C10(v7, v6);
  v9 = (__int64)v8;
  if ( v8 )
  {
    if ( (unsigned __int8)sub_14A6CB0(a1, a2, (__int64)v8, a3, v12) )
      return v12[0];
    v10 = sub_14A6CB0(a2, a1, v9, a3, v12);
    if ( (_BYTE)v10 )
    {
      return v12[0];
    }
    else if ( a3 )
    {
      *a3 = sub_14A6B70(v9);
    }
  }
  else
  {
    v10 = 1;
    if ( a3 )
      *a3 = 0;
  }
  return v10;
}
