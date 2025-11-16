// Function: sub_1984400
// Address: 0x1984400
//
bool __fastcall sub_1984400(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r12
  bool result; // al
  __int64 v6; // r15
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 *v18[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v19[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = sub_146F1B0(*(_QWORD *)(a1 + 16), *(_QWORD *)a2);
  result = 0;
  if ( *(_WORD *)(v4 + 24) == 7 )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(_DWORD *)(a2 + 16) + 1;
    v8 = sub_146F1B0(v6, **(_QWORD **)(a2 + 8));
    v9 = sub_14806B0(v6, v8, v4, 0, 0);
    v10 = *(_QWORD *)(a1 + 16);
    v11 = v9;
    v12 = sub_1456040(v9);
    v13 = sub_145CF80(v10, v12, v7, 0);
    v14 = sub_13A5BC0((_QWORD *)v4, *(_QWORD *)(a1 + 16));
    v15 = *(_QWORD **)(a1 + 16);
    v16 = v14;
    v18[0] = v19;
    v19[0] = v11;
    v19[1] = v13;
    v18[1] = (__int64 *)0x200000002LL;
    v17 = sub_147EE30(v15, v18, 0, 0, a3, a4);
    if ( v18[0] != v19 )
      _libc_free((unsigned __int64)v18[0]);
    return v16 == v17;
  }
  return result;
}
