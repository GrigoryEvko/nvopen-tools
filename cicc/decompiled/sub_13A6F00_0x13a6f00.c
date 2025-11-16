// Function: sub_13A6F00
// Address: 0x13a6f00
//
__int64 __fastcall sub_13A6F00(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 *a4)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  _QWORD *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v6 = a2;
  if ( *(_WORD *)(a2 + 24) != 7 )
    return sub_13A6BC0(a1, v6, a3);
  while ( 1 )
  {
    v7 = v6;
    v8 = (_QWORD *)v6;
    v6 = **(_QWORD **)(v6 + 32);
    v19 = sub_13A5BC0(v8, *(_QWORD *)(a1 + 8));
    v9 = sub_1481F60(*(_QWORD *)(a1 + 8), *(_QWORD *)(v7 + 48));
    if ( !(unsigned __int8)sub_14562D0(v9) )
    {
      v16 = *(_QWORD *)(a1 + 8);
      v10 = sub_1456040(v6);
      v17 = sub_1456C90(v16, v10);
      v15 = *(_QWORD *)(a1 + 8);
      v11 = sub_1456040(v9);
      if ( v17 < sub_1456C90(v15, v11) && (*(_BYTE *)(v7 + 26) & 7) == 0 )
        break;
    }
    if ( !(unsigned __int8)sub_13A6BC0(a1, v19, a3) )
      break;
    v12 = sub_13A6B70(a1, *(_QWORD ***)(v7 + 48));
    v13 = *a4;
    if ( (*a4 & 1) != 0 )
      *a4 = 2 * ((v13 >> 58 << 57) | ~(-1LL << (v13 >> 58)) & (~(-1LL << (v13 >> 58)) & (v13 >> 1) | (1LL << v12))) + 1;
    else
      *(_QWORD *)(*(_QWORD *)v13 + 8LL * (v12 >> 6)) |= 1LL << v12;
    if ( *(_WORD *)(v6 + 24) != 7 )
      return sub_13A6BC0(a1, v6, a3);
  }
  return 0;
}
