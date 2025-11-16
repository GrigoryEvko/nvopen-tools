// Function: sub_19E5AB0
// Address: 0x19e5ab0
//
_QWORD *__fastcall sub_19E5AB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 *v8; // rdi
  _QWORD *v9; // r12
  bool v10; // cc
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 ***v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax

  v6 = sub_19E1ED0(a1, *(__int64 ****)(a2 - 48));
  v7 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 64), 72, 16);
  v8 = (__int64 *)(a1 + 64);
  v9 = v7;
  v7[1] = 0xFFFFFFFD0000000CLL;
  v7[2] = 0;
  v7[6] = a3;
  v10 = *(_DWORD *)(a1 + 176) <= 1u;
  v7[3] = 0;
  v7[4] = 2;
  v7[5] = 0;
  *v7 = &unk_49F4E50;
  v7[7] = a2;
  v7[8] = v6;
  if ( v10 || (v11 = *(_QWORD *)(a1 + 168), (v12 = *(_QWORD **)(v11 + 8)) == 0) )
    v12 = (_QWORD *)sub_145CBF0(v8, 16, 8);
  else
    *(_QWORD *)(v11 + 8) = *v12;
  v9[3] = v12;
  v13 = *(__int64 ****)(a2 - 24);
  v14 = **(_QWORD **)(a2 - 48);
  *((_DWORD *)v9 + 3) = 0;
  v9[5] = v14;
  v15 = sub_19E1ED0(a1, v13);
  v16 = v9[3];
  v17 = v15;
  v18 = *((unsigned int *)v9 + 9);
  *((_DWORD *)v9 + 9) = v18 + 1;
  *(_QWORD *)(v16 + 8 * v18) = v17;
  return v9;
}
