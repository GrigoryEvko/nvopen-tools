// Function: sub_318D720
// Address: 0x318d720
//
void __fastcall sub_318D720(__int64 a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rdi
  _QWORD v7[2]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+10h] [rbp-30h]
  __int64 v9; // [rsp+18h] [rbp-28h]

  v1 = *(_QWORD **)(a1 + 8);
  v2 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 || !v2 )
  {
    v4 = *(_QWORD *)(v2 + 24);
    v5 = *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL;
    v6 = *(_QWORD **)(a1 + 8);
    v7[0] = *(_QWORD *)(v2 + 16);
    v7[1] = v7[0] + 48LL;
    v9 = v4;
    v8 = 0;
    sub_318CB50(v6, v5, (__int64)v7);
  }
  else
  {
    sub_318B480((__int64)v7, *(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
    v3 = sub_318B4F0(v2);
    sub_318CB50(v1, v3, (__int64)v7);
  }
}
