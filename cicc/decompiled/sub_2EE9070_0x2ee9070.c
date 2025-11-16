// Function: sub_2EE9070
// Address: 0x2ee9070
//
void __fastcall sub_2EE9070(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdi

  v7 = *(int *)(a2 + 24);
  *(_DWORD *)(a1[42] + 8 * v7) = -1;
  v8 = a1[50];
  if ( v8 )
    sub_2EE8CA0(v8, a2, v7, a4, a5, a6);
  v9 = a1[51];
  if ( v9 )
    sub_2EE8CA0(v9, a2, v7, a4, a5, a6);
}
