// Function: sub_D344F0
// Address: 0xd344f0
//
__int64 __fastcall sub_D344F0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, char a6, __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]

  v9 = sub_D326F0(a3, *(_QWORD *)(a1 + 8), a7);
  if ( !v9 )
    return 0;
  v15 = v9;
  v10 = sub_D326F0(a4, *(_QWORD *)a1, a7);
  if ( !v10 )
    return 0;
  if ( a3 == v15 )
  {
    *(_QWORD *)(a1 + 8) = a3;
    if ( a4 == v10 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( a4 != v10 )
LABEL_5:
    *(_QWORD *)a1 = a4;
LABEL_6:
  v13 = *(unsigned int *)(a1 + 24);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v13 + 1, 4u, v11, v12);
    v13 = *(unsigned int *)(a1 + 24);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v13) = a2;
  ++*(_DWORD *)(a1 + 24);
  *(_BYTE *)(a1 + 44) |= a6;
  return 1;
}
