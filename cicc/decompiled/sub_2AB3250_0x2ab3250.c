// Function: sub_2AB3250
// Address: 0x2ab3250
//
__int64 __fastcall sub_2AB3250(__int64 *a1, void *a2, void *a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  void *v15[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-30h]

  v5 = sub_D4B130(a1[1]);
  v15[0] = a2;
  v6 = a1[3];
  a1[30] = v5;
  v7 = v5;
  v15[2] = "scalar.ph";
  v15[1] = a3;
  v8 = a1[4];
  v16 = 773;
  v9 = sub_986580(v5);
  v10 = sub_F36960(v7, (__int64 *)(v9 + 24), 0, v8, v6, 0, v15, 0);
  v11 = 0;
  a1[31] = v10;
  v12 = v10;
  v13 = *(_QWORD *)(a1[58] + 8);
  if ( *(_DWORD *)(v13 + 64) == 1 )
    v11 = **(_QWORD **)(v13 + 56);
  return sub_2AB1A90(v11, v12);
}
