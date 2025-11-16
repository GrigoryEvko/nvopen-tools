// Function: sub_1A2F320
// Address: 0x1a2f320
//
__int64 __fastcall sub_1A2F320(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r13
  _QWORD *v7; // rsi
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD *)(a1 + 32);
  v11[0] = a2;
  sub_1A2EDE0(v3 + 208, v11);
  v4 = *(_QWORD *)(a1 + 128);
  if ( v4 == *(_QWORD *)(a1 + 56) )
  {
    v5 = *(_QWORD *)(a1 + 136);
    if ( v5 == *(_QWORD *)(a1 + 64) )
    {
      v6 = sub_159C470(**(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v5 - v4, 0);
      v7 = (_QWORD *)sub_1A246E0((__int64 *)a1, a1 + 192, **(_QWORD **)(a1 + 168));
      v8 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v8 + 16) )
        BUG();
      v9 = (__int64 *)(a1 + 192);
      if ( *(_DWORD *)(v8 + 36) == 117 )
        sub_15E7DE0(v9, v7, v6);
      else
        sub_15E7E90(v9, v7, v6);
    }
  }
  return 1;
}
