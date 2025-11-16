// Function: sub_2C1B990
// Address: 0x2c1b990
//
_QWORD *__fastcall sub_2C1B990(__int64 a1)
{
  unsigned __int8 *v1; // r8
  _QWORD *v2; // r15
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r9
  _QWORD *v6; // r12
  __int64 v7; // rbx
  unsigned __int8 *v9; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v10; // [rsp+8h] [rbp-48h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(unsigned __int8 **)(a1 + 136);
  v2 = *(_QWORD **)(a1 + 48);
  v3 = *(unsigned int *)(a1 + 56);
  v11[0] = *(_QWORD *)(a1 + 88);
  if ( v11[0] )
  {
    v9 = v1;
    sub_2AAAFA0(v11);
    v1 = v9;
  }
  v10 = v1;
  v4 = sub_22077B0(0xA8u);
  v6 = (_QWORD *)v4;
  if ( v4 )
  {
    v7 = *(_QWORD *)(a1 + 160);
    sub_2ABAD10(v4, 14, v2, v3, v10, v5);
    v6[20] = v7;
    *v6 = &unk_4A23C98;
    v6[5] = &unk_4A23CD0;
    v6[12] = &unk_4A23D08;
  }
  sub_9C6650(v11);
  return v6;
}
