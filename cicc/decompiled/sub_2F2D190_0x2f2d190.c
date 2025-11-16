// Function: sub_2F2D190
// Address: 0x2f2d190
//
__int64 __fastcall sub_2F2D190(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r12
  bool v5; // r8
  __int64 v6; // r13
  char v8; // [rsp+Ch] [rbp-34h]

  v2 = sub_2DCBDB0(a1, a2);
  if ( !v3 )
    return v2;
  v4 = (_QWORD *)v3;
  v5 = 1;
  if ( !v2 && v3 != a1 + 8 )
    v5 = *a2 < *(_DWORD *)(v3 + 32);
  v8 = v5;
  v6 = sub_22077B0(0x28u);
  *(_DWORD *)(v6 + 32) = *a2;
  sub_220F040(v8, v6, v4, (_QWORD *)(a1 + 8));
  ++*(_QWORD *)(a1 + 40);
  return v6;
}
