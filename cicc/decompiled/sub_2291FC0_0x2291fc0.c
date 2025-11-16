// Function: sub_2291FC0
// Address: 0x2291fc0
//
_QWORD *__fastcall sub_2291FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  _QWORD *v16; // r13
  unsigned int v17; // [rsp+Ch] [rbp-64h]
  __int64 v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  unsigned __int64 v20[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v21[8]; // [rsp+30h] [rbp-40h] BYREF

  v7 = *(__int64 **)(a1 + 8);
  if ( *(_WORD *)(a2 + 24) != 8 )
  {
    v11 = a3;
    v12 = a4;
    v10 = *(_QWORD *)(a1 + 8);
    return sub_DC1960(v10, a2, v12, v11, 0);
  }
  if ( a3 != *(_QWORD *)(a2 + 48) )
  {
    if ( !sub_DADE90(*(_QWORD *)(a1 + 8), a2, a3) )
    {
      v19 = *(_QWORD *)(a1 + 8);
      v17 = *(_WORD *)(a2 + 28) & 7;
      v18 = *(_QWORD *)(a2 + 48);
      v14 = sub_D33D80((_QWORD *)a2, v19, v8, v9, v17);
      v15 = sub_2291FC0(a1, **(_QWORD **)(a2 + 32), a3, a4);
      return sub_DC1960(v19, v15, v14, v18, v17);
    }
    v10 = *(_QWORD *)(a1 + 8);
    v11 = a3;
    v12 = a4;
    return sub_DC1960(v10, a2, v12, v11, 0);
  }
  v21[0] = sub_D33D80((_QWORD *)a2, *(_QWORD *)(a1 + 8), a3, a4, a5);
  v20[0] = (unsigned __int64)v21;
  v21[1] = a4;
  v20[1] = 0x200000002LL;
  v16 = sub_DC7EB0(v7, (__int64)v20, 0, 0);
  if ( (_QWORD *)v20[0] != v21 )
    _libc_free(v20[0]);
  if ( sub_D968A0((__int64)v16) )
    return **(_QWORD ***)(a2 + 32);
  else
    return sub_DC1960(
             *(_QWORD *)(a1 + 8),
             **(_QWORD **)(a2 + 32),
             (__int64)v16,
             *(_QWORD *)(a2 + 48),
             *(_WORD *)(a2 + 28) & 7);
}
