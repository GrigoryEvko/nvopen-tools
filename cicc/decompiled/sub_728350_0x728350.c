// Function: sub_728350
// Address: 0x728350
//
_QWORD *__fastcall sub_728350(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // r13
  _QWORD *result; // rax
  int v6; // ecx
  _QWORD *i; // rbx
  __int64 v8; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 112);
  v3 = *(_QWORD *)(a1 + 168);
  v4 = *(_QWORD *)(a1 + 264);
  v8 = *(_QWORD *)(a1 + 104);
  result = (_QWORD *)(v4 | v3 | v2 | v8);
  if ( result )
  {
    result = sub_724930(a2, *(_DWORD *)(a1 + 24));
    result[1] = a2;
    v6 = *(_DWORD *)(a1 + 24);
    result[3] = v8;
    *((_DWORD *)result + 4) = v6;
    result[4] = v2;
    result[5] = v3;
    result[6] = v4;
    *result = 0;
    if ( unk_4F072C0 )
      **(_QWORD **)(unk_4D03FF0 + 352LL) = result;
    else
      unk_4F072C0 = result;
    *(_QWORD *)(unk_4D03FF0 + 352LL) = result;
    *(_BYTE *)(a1 + 29) |= 4u;
  }
  for ( i = *(_QWORD **)(a1 + 160); i; i = (_QWORD *)*i )
    result = (_QWORD *)sub_728350(i, a2);
  return result;
}
