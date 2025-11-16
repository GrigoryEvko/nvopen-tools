// Function: sub_1DB53F0
// Address: 0x1db53f0
//
__int64 __fastcall sub_1DB53F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  __int64 i; // rbx
  _QWORD *v11; // rdx
  _BYTE v13[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v15)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  v3 = *(unsigned int *)(a1 + 112);
  sub_1F4AA00(v13, v3, 0, 0, 0);
  if ( !v14 )
    sub_4263D6(v13, v3, v4);
  v15(v13, a2);
  v5 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 16) )
  {
    sub_16E7DE0(a2, 32);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v5 + 1;
    *v5 = 32;
  }
  if ( v14 )
    v14(v13, v13, 3);
  sub_1DB50D0(a1, a2);
  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 104) )
    sub_1DB5300(i, a2, v6, v7, v8, v9);
  v11 = *(_QWORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v11 <= 7u )
  {
    a2 = sub_16E7EE0(a2, " weight:", 8u);
  }
  else
  {
    *v11 = 0x3A74686769657720LL;
    *(_QWORD *)(a2 + 24) += 8LL;
  }
  return sub_16E7B70(a2);
}
