// Function: sub_2C1BB80
// Address: 0x2c1bb80
//
_QWORD *__fastcall sub_2C1BB80(__int64 a1)
{
  _QWORD *v1; // r13
  unsigned __int8 *v2; // rbx
  _QWORD *v3; // r14
  __int64 v4; // rax
  __int64 v5; // r9
  _QWORD *v6; // r12

  v1 = *(_QWORD **)(a1 + 48);
  v2 = *(unsigned __int8 **)(a1 + 136);
  v3 = &v1[*(unsigned int *)(a1 + 56)];
  v4 = sub_22077B0(0xA0u);
  v6 = (_QWORD *)v4;
  if ( v4 )
  {
    sub_2ABDBC0(v4, 24, v1, v3, v2, v5);
    *v6 = &unk_4A23E30;
    v6[5] = &unk_4A23E70;
    v6[12] = &unk_4A23EA8;
  }
  return v6;
}
