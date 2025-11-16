// Function: sub_2789F00
// Address: 0x2789f00
//
__int64 __fastcall sub_2789F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbp
  __int64 v5; // rax
  __int64 result; // rax
  unsigned __int8 v7; // [rsp-39h] [rbp-39h]
  __int64 v8; // [rsp-38h] [rbp-38h] BYREF
  _QWORD *v9; // [rsp-30h] [rbp-30h]
  __int64 v10; // [rsp-28h] [rbp-28h]
  int v11; // [rsp-20h] [rbp-20h]
  char v12; // [rsp-1Ch] [rbp-1Ch]
  _QWORD v13[3]; // [rsp-18h] [rbp-18h] BYREF

  v5 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)(a1 + 40) == v5 )
    return sub_B19DB0(a4, a1, a2);
  v13[2] = v4;
  v10 = 0x100000001LL;
  v9 = v13;
  v11 = 0;
  v12 = 1;
  v13[0] = v5;
  v8 = 1;
  result = (unsigned int)sub_D0EBA0(a1, a3, (__int64)&v8, a4, 0) ^ 1;
  if ( !v12 )
  {
    v7 = result;
    _libc_free((unsigned __int64)v9);
    return v7;
  }
  return result;
}
