// Function: sub_2D112B0
// Address: 0x2d112b0
//
_QWORD *__fastcall sub_2D112B0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r13
  bool v5; // r8
  __int64 v6; // rbx
  char v7; // [rsp+Ch] [rbp-34h]

  result = sub_2D11210(a1, a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != (_QWORD *)(a1 + 8) )
      v5 = *a2 < v3[4];
    v7 = v5;
    v6 = sub_22077B0(0x28u);
    *(_QWORD *)(v6 + 32) = *a2;
    sub_220F040(v7, v6, v4, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
    return (_QWORD *)v6;
  }
  return result;
}
