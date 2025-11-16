// Function: sub_1B970A0
// Address: 0x1b970a0
//
__int64 __fastcall sub_1B970A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rbx
  __int64 v7; // r15
  unsigned int v9; // r14d
  _QWORD *v10; // rax
  __int64 *v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]

  v6 = (__int64 *)(a1 + 96);
  v14 = *(_QWORD *)(a3 + 32);
  v7 = *(_QWORD *)(*(_QWORD *)a2 + 24LL);
  if ( sub_15FBDD0(v7, *(_QWORD *)(a3 + 24), a4) )
  {
    v16 = 257;
    return sub_17FE280(v6, a2, a3, v15);
  }
  else
  {
    v9 = sub_127FA20(a4, v7);
    v10 = (_QWORD *)sub_16498A0(a2);
    v11 = (__int64 *)sub_1644C60(v10, v9);
    v12 = sub_16463B0(v11, v14);
    v16 = 257;
    v13 = sub_17FE280(v6, a2, (__int64)v12, v15);
    v16 = 257;
    return sub_17FE280(v6, v13, a3, v15);
  }
}
