// Function: sub_356DE20
// Address: 0x356de20
//
__int64 __fastcall sub_356DE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 *v10; // rsi
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 v13; // [rsp+8h] [rbp-28h]
  __int64 v14; // [rsp+10h] [rbp-20h]
  unsigned int v15; // [rsp+18h] [rbp-18h]

  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  sub_356D680(a1, a2, (__int64)&v12, a4, a5, a6);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(a2 + 328);
  if ( v7 )
  {
    v8 = (unsigned int)(*(_DWORD *)(v7 + 24) + 1);
    v9 = *(_DWORD *)(v7 + 24) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  v10 = 0;
  if ( v9 < *(_DWORD *)(v6 + 32) )
    v10 = *(__int64 **)(*(_QWORD *)(v6 + 24) + 8 * v8);
  sub_356DB60(a1, v10, *(_QWORD **)(a1 + 32));
  return sub_C7D6A0(v13, 16LL * v15, 8);
}
