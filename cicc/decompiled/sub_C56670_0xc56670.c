// Function: sub_C56670
// Address: 0xc56670
//
__int64 __fastcall sub_C56670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rdi
  __int64 result; // rax
  _QWORD *v18; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 v19; // [rsp+8h] [rbp-88h]
  _QWORD v20[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v21[14]; // [rsp+20h] [rbp-70h] BYREF

  sub_C54F20(a1, a2, a5);
  LOBYTE(v20[0]) = 0;
  v18 = v20;
  v21[5] = 0x100000000LL;
  v21[6] = &v18;
  v19 = 0;
  memset(&v21[1], 0, 32);
  v21[0] = &unk_49DD210;
  sub_CB5980(v21, 0, 0, 0);
  sub_CB59F0(v21, a3);
  v21[0] = &unk_49DD210;
  sub_CB5840(v21);
  v7 = sub_CB7210(v21);
  v8 = *(_WORD **)(v7 + 32);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_CB6200(v7, "= ", 2);
  }
  else
  {
    *v8 = 8253;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v10 = 0;
  sub_CB6200(v9, v18, v19);
  if ( v19 < 8 )
    v10 = 8 - v19;
  v11 = sub_CB7210(v9);
  v12 = sub_CB69B0(v11, v10);
  sub_904010(v12, " (default: ");
  if ( *(_BYTE *)(a4 + 16) )
  {
    v13 = sub_CB7210(v12);
    sub_CB59F0(v13, *(_QWORD *)(a4 + 8));
  }
  else
  {
    v13 = sub_CB7210(v12);
    sub_904010(v13, "*no default*");
  }
  v14 = sub_CB7210(v13);
  v15 = *(_WORD **)(v14 + 32);
  v16 = v14;
  if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 1u )
  {
    result = sub_CB6200(v14, ")\n", 2);
  }
  else
  {
    result = 2601;
    *v15 = 2601;
    *(_QWORD *)(v16 + 32) += 2LL;
  }
  if ( v18 != v20 )
    return j_j___libc_free_0(v18, v20[0] + 1LL);
  return result;
}
