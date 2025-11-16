// Function: sub_2E0B730
// Address: 0x2e0b730
//
__int64 __fastcall sub_2E0B730(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 i; // rbx
  __int64 v7; // rdx
  _BYTE v9[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v10)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v11)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  v3 = *(unsigned int *)(a1 + 112);
  sub_2FF6320(v9, v3, 0, 0, 0);
  if ( !v10 )
    sub_4263D6(v9, v3, v4);
  v11(v9, a2);
  v5 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 24) )
  {
    sub_CB5D20(a2, 32);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v5 + 1;
    *v5 = 32;
  }
  if ( v10 )
    v10(v9, v9, 3);
  sub_2E0B3F0(a1, a2);
  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 104) )
    sub_2E0B620(i, a2);
  v7 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v7) <= 8 )
  {
    a2 = sub_CB6200(a2, "  weight:", 9u);
  }
  else
  {
    *(_BYTE *)(v7 + 8) = 58;
    *(_QWORD *)v7 = 0x7468676965772020LL;
    *(_QWORD *)(a2 + 32) += 9LL;
  }
  return sub_CB5AB0(a2, *(float *)(a1 + 116));
}
