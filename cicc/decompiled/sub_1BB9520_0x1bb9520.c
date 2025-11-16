// Function: sub_1BB9520
// Address: 0x1bb9520
//
__int64 __fastcall sub_1BB9520(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BOOL4 v4; // r8d
  __int64 v5; // r15
  __int64 v8; // r12
  _BOOL4 v10; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = a1 + 8;
  if ( !a2 && a3 != v5 )
    v4 = *(_DWORD *)(*(_QWORD *)(a3 + 32) + 84LL) < *(_DWORD *)(*(_QWORD *)a4 + 84LL);
  v10 = v4;
  v8 = sub_22077B0(40);
  *(_QWORD *)(v8 + 32) = *(_QWORD *)a4;
  sub_220F040(v10, v8, a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
