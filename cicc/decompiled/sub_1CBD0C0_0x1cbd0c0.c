// Function: sub_1CBD0C0
// Address: 0x1cbd0c0
//
_QWORD *__fastcall sub_1CBD0C0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  _BOOL4 v5; // r14d
  __int64 v6; // rax
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v7[0] = a2;
  result = sub_1CBCEE0(a1 + 112, v7);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != a1 + 120 )
      v5 = v7[0] < *(_QWORD *)(v3 + 32);
    v6 = sub_22077B0(40);
    *(_QWORD *)(v6 + 32) = v7[0];
    result = (_QWORD *)sub_220F040(v5, v6, v4, a1 + 120);
    ++*(_QWORD *)(a1 + 152);
  }
  return result;
}
