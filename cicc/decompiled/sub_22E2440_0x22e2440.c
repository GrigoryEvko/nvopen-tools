// Function: sub_22E2440
// Address: 0x22e2440
//
__int64 __fastcall sub_22E2440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // rsi
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h]
  unsigned int v14; // [rsp+18h] [rbp-18h]

  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  sub_22E1CA0(a1, a2, (__int64)&v11, a4, a5, a6);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v7 = (unsigned int)(*(_DWORD *)(v6 + 20) + 1);
    LODWORD(v6) = *(_DWORD *)(v6 + 20) + 1;
  }
  v9 = 0;
  if ( (unsigned int)v6 < *(_DWORD *)(v8 + 32) )
    v9 = *(__int64 **)(*(_QWORD *)(v8 + 24) + 8 * v7);
  sub_22E2180(a1, v9, *(_QWORD **)(a1 + 32));
  return sub_C7D6A0(v12, 16LL * v14, 8);
}
