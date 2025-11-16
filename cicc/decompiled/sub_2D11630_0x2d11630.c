// Function: sub_2D11630
// Address: 0x2d11630
//
void *__fastcall sub_2D11630(__int64 a1, unsigned __int64 a2)
{
  void *result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r12
  char v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v7[0] = a2;
  result = sub_2D11450(a1 + 40, v7);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != (_QWORD *)(a1 + 48) )
      v5 = v7[0] < v3[4];
    v6 = sub_22077B0(0x28u);
    *(_QWORD *)(v6 + 32) = v7[0];
    result = sub_220F040(v5, v6, v4, (_QWORD *)(a1 + 48));
    ++*(_QWORD *)(a1 + 80);
  }
  return result;
}
