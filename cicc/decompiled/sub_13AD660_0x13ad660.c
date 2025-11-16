// Function: sub_13AD660
// Address: 0x13ad660
//
__int64 __fastcall sub_13AD660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned int v14; // [rsp+Ch] [rbp-64h]
  __int64 v15; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+18h] [rbp-58h]
  unsigned __int64 v17[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v18[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_WORD *)(a2 + 24) != 7 )
  {
    v8 = a3;
    v9 = a4;
    v7 = *(_QWORD *)(a1 + 8);
    return sub_14799E0(v7, a2, v9, v8, 0);
  }
  if ( a3 != *(_QWORD *)(a2 + 48) )
  {
    if ( !(unsigned __int8)sub_146CEE0(*(_QWORD *)(a1 + 8), a2, a3) )
    {
      v16 = *(_QWORD *)(a1 + 8);
      v14 = *(_WORD *)(a2 + 26) & 7;
      v15 = *(_QWORD *)(a2 + 48);
      v11 = sub_13A5BC0((_QWORD *)a2, v16);
      v12 = sub_13AD660(a1, **(_QWORD **)(a2 + 32), a3, a4);
      return sub_14799E0(v16, v12, v11, v15, v14);
    }
    v7 = *(_QWORD *)(a1 + 8);
    v8 = a3;
    v9 = a4;
    return sub_14799E0(v7, a2, v9, v8, 0);
  }
  v18[0] = sub_13A5BC0((_QWORD *)a2, *(_QWORD *)(a1 + 8));
  v17[0] = (unsigned __int64)v18;
  v18[1] = a4;
  v17[1] = 0x200000002LL;
  v13 = sub_147DD40(v6, v17, 0, 0);
  if ( (_QWORD *)v17[0] != v18 )
    _libc_free(v17[0]);
  if ( (unsigned __int8)sub_14560B0(v13) )
    return **(_QWORD **)(a2 + 32);
  else
    return sub_14799E0(*(_QWORD *)(a1 + 8), **(_QWORD **)(a2 + 32), v13, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
}
