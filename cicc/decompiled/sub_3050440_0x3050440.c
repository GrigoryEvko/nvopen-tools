// Function: sub_3050440
// Address: 0x3050440
//
__int64 __fastcall sub_3050440(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rdx
  bool v11; // cc
  _QWORD *v12; // rax
  __int64 v13; // r13
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  int v16; // [rsp+8h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v15 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v15, v8, 1);
  v10 = *(_QWORD *)(v9 + 96);
  v11 = *(_DWORD *)(v10 + 32) <= 0x40u;
  v16 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD **)(v10 + 24);
  if ( !v11 )
    v12 = (_QWORD *)*v12;
  if ( (_DWORD)v12 == 8170 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  }
  else if ( (_DWORD)v12 == 8455 )
  {
    v13 = sub_304FBD0(a1, a2, a3, a4);
  }
  else
  {
    v13 = 0;
  }
  if ( v15 )
    sub_B91220((__int64)&v15, v15);
  return v13;
}
