// Function: sub_357C320
// Address: 0x357c320
//
__int64 __fastcall sub_357C320(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rsi
  unsigned __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501FE44 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_17;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501FE44);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 200;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_501FE3C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_16;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_501FE3C);
  sub_357C2D0(v14, a2, (__int64 *)(v10 + 208), v7, 1);
  v11 = v14[0];
  v12 = *(_QWORD *)(a1 + 200);
  v14[0] = 0;
  *(_QWORD *)(a1 + 200) = v11;
  if ( v12 )
  {
    sub_3575560(a1 + 200, v12);
    if ( v14[0] )
      sub_3575560((__int64)v14, v14[0]);
  }
  return 0;
}
