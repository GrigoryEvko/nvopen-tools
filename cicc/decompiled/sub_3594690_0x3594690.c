// Function: sub_3594690
// Address: 0x3594690
//
__int64 __fastcall sub_3594690(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi

  v3 = *a2;
  v4 = sub_22077B0(0x38u);
  v5 = v4;
  if ( v4 )
    sub_35933D0(v4, v3);
  v6 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v5;
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  return 0;
}
