// Function: sub_2CE14D0
// Address: 0x2ce14d0
//
_QWORD *__fastcall sub_2CE14D0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r12
  bool v5; // r8
  __int64 v6; // r13
  char v8; // [rsp+Ch] [rbp-34h]

  v2 = sub_23FDE00(a1, a2);
  if ( !v3 )
    return v2;
  v4 = v3;
  v5 = 1;
  if ( !v2 && v3 != (_QWORD *)(a1 + 8) )
    v5 = *a2 < v3[4];
  v8 = v5;
  v6 = sub_22077B0(0x28u);
  *(_QWORD *)(v6 + 32) = *a2;
  sub_220F040(v8, v6, v4, (_QWORD *)(a1 + 8));
  ++*(_QWORD *)(a1 + 40);
  return (_QWORD *)v6;
}
