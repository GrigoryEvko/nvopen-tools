// Function: sub_6891A0
// Address: 0x6891a0
//
__int64 sub_6891A0()
{
  __int64 result; // rax
  _QWORD *v1; // rbx
  __int64 v2; // rdi
  _QWORD *v3; // r12
  __int64 v4; // rax

  result = unk_4D03C50;
  v1 = *(_QWORD **)(unk_4D03C50 + 32LL);
  *(_BYTE *)(unk_4D03C50 + 18LL) &= ~4u;
  for ( *(_QWORD *)(result + 32) = 0; v1; result = sub_6E1D00(v3) )
  {
    v2 = v1[1];
    v3 = v1;
    v1 = (_QWORD *)*v1;
    v4 = *(_QWORD *)(v2 + 16);
    if ( v4 )
      sub_6EB360(v2, *(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL), *(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL), v3 + 2);
  }
  return result;
}
