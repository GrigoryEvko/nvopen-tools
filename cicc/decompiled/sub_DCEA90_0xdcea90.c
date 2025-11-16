// Function: sub_DCEA90
// Address: 0xdcea90
//
unsigned __int64 __fastcall sub_DCEA90(__int64 a1, unsigned __int64 a2)
{
  unsigned __int16 v3; // r13
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 *v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // [rsp+10h] [rbp-80h]
  _BYTE *v12; // [rsp+20h] [rbp-70h] BYREF
  __int64 v13; // [rsp+28h] [rbp-68h]
  _BYTE v14[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = *(_WORD *)(a2 + 24);
  if ( v3 != *(_WORD *)(a1 + 8) && v3 != *(_WORD *)(a1 + 10) )
    return a2;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 32);
  v12 = v14;
  v13 = 0x600000000LL;
  if ( (unsigned __int8)sub_DCEB80(a1, v6, v5, &v12) )
  {
    if ( (_DWORD)v13 )
    {
      v9 = *(__int64 **)a1;
      v6 = v3;
      if ( *(_WORD *)(a2 + 24) == 13 )
        v10 = (unsigned __int64)sub_DCEA30(v9, v3, (__int64)&v12, v7, v8);
      else
        v10 = sub_DCD310(v9, v3, (__int64)&v12, v7, v8);
      v11 = v10;
    }
  }
  else
  {
    v11 = a2;
  }
  if ( v12 != v14 )
    _libc_free(v12, v6);
  return v11;
}
