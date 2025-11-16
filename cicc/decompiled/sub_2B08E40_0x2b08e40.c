// Function: sub_2B08E40
// Address: 0x2b08e40
//
__int64 __fastcall sub_2B08E40(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __int64 v8; // r12
  char v10; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && a3 != v5 )
    v4 = *(_DWORD *)(a3[4] + 140LL) < *(_DWORD *)(*(_QWORD *)a4 + 140LL);
  v10 = v4;
  v8 = sub_22077B0(0x28u);
  *(_QWORD *)(v8 + 32) = *(_QWORD *)a4;
  sub_220F040(v10, v8, a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
