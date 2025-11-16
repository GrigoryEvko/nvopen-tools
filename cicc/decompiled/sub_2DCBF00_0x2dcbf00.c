// Function: sub_2DCBF00
// Address: 0x2dcbf00
//
__int64 __fastcall sub_2DCBF00(__int64 a1, unsigned int *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r13
  bool v5; // r8
  __int64 v6; // rbx
  char v7; // [rsp+Ch] [rbp-34h]

  result = sub_2DCBDB0(a1, a2);
  if ( v3 )
  {
    v4 = (_QWORD *)v3;
    v5 = 1;
    if ( !result && v3 != a1 + 8 )
      v5 = *a2 < *(_DWORD *)(v3 + 32);
    v7 = v5;
    v6 = sub_22077B0(0x28u);
    *(_DWORD *)(v6 + 32) = *a2;
    sub_220F040(v7, v6, v4, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
    return v6;
  }
  return result;
}
