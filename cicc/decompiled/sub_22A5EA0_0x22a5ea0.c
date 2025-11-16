// Function: sub_22A5EA0
// Address: 0x22a5ea0
//
__int64 __fastcall sub_22A5EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r15
  int v8; // eax
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // eax
  unsigned __int64 *v14; // rdx

  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0x100000000LL;
  v4 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = **(_QWORD **)(v4 + 8);
  v8 = *(_DWORD *)(a1 + 44);
  *(_DWORD *)(a1 + 40) = 0;
  if ( v8 )
  {
    v9 = 0;
  }
  else
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), 1u, 8u, v5, v6);
    v9 = 8LL * *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + v9) = v7;
  v10 = *(__int64 **)(a1 + 32);
  ++*(_DWORD *)(a1 + 40);
  v11 = *v10;
  if ( v11 )
  {
    v12 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
    v13 = *(_DWORD *)(v11 + 44) + 1;
  }
  else
  {
    v12 = 0;
    v13 = 0;
  }
  v14 = 0;
  if ( v13 < *(_DWORD *)(v4 + 40) )
    v14 = *(unsigned __int64 **)(*(_QWORD *)(v4 + 32) + 8 * v12);
  sub_22A5210(a1, v4 + 8, v14);
  return a1;
}
