// Function: sub_28AF6D0
// Address: 0x28af6d0
//
__int64 __fastcall sub_28AF6D0(__int64 a1, __int64 a2, __int64 a3, _QWORD **a4)
{
  unsigned __int8 *v7; // rbx
  unsigned __int8 *v8; // rax
  _QWORD *v9; // rdi
  _QWORD v11[6]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v12[12]; // [rsp+30h] [rbp-60h] BYREF

  v7 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  v8 = sub_BD3990(*(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), a2);
  v9 = *a4;
  v12[0] = v7;
  v12[1] = 1;
  memset(&v12[2], 0, 32);
  v11[0] = v8;
  v11[1] = 1;
  memset(&v11[2], 0, 32);
  if ( (unsigned __int8)sub_CF4D50((__int64)v9, (__int64)v11, (__int64)v12, (__int64)(a4 + 1), 0) == 3 )
    return sub_28AE7F0(a1, a2, a3, a4);
  else
    return 0;
}
