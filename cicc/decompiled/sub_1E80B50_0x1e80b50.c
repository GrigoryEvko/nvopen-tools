// Function: sub_1E80B50
// Address: 0x1e80b50
//
void __fastcall sub_1E80B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rdi

  v6 = *(int *)(a2 + 48);
  *(_DWORD *)(*(_QWORD *)(a1 + 552) + 8 * v6) = -1;
  v7 = *(_QWORD *)(a1 + 616);
  if ( v7 )
    sub_1E807B0(v7, a2, v6, a4, a5, a6);
}
