// Function: sub_15940C0
// Address: 0x15940c0
//
__int64 __fastcall sub_15940C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi

  sub_1648CB0(a1, a2, 14);
  *(_DWORD *)(a1 + 20) &= 0xF0000000;
  v6 = sub_16982C0(a1, a2, v4, v5);
  v7 = a3 + 8;
  v8 = a1 + 32;
  if ( *(_QWORD *)(a3 + 8) == v6 )
    return sub_169C6E0(v8, v7);
  else
    return sub_16986C0(v8, v7);
}
