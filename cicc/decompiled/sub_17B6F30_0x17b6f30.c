// Function: sub_17B6F30
// Address: 0x17b6f30
//
__int64 __fastcall sub_17B6F30(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  char v12; // dl
  _QWORD v14[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = sub_16471D0(*(_QWORD **)(a1 + 64), 0);
  v3 = *(_QWORD **)(a1 + 64);
  v14[0] = v2;
  v4 = sub_16471D0(v3, 0);
  v5 = *(_QWORD **)(a1 + 64);
  v14[1] = v4;
  v6 = sub_1643350(v5);
  v7 = *(_QWORD **)(a1 + 64);
  v14[2] = v6;
  v8 = (__int64 *)sub_1643270(v7);
  v9 = sub_1644EA0(v8, v14, 3, 0);
  v10 = sub_1632190(*(_QWORD *)(a1 + 48), (__int64)"llvm_gcda_start_file", 20, v9);
  if ( *(_BYTE *)(v10 + 16) )
    return v10;
  v11 = **(_QWORD **)(a1 + 56);
  if ( *(_BYTE *)(v11 + 144) )
  {
    v12 = 58;
  }
  else
  {
    v12 = 40;
    if ( !*(_BYTE *)(v11 + 146) )
      return v10;
  }
  sub_15E0DF0(v10, 2, v12);
  return v10;
}
