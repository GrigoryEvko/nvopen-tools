// Function: sub_3952A40
// Address: 0x3952a40
//
__int64 __fastcall sub_3952A40(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 *v6; // rbx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // r12

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9E06C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_19;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9E06C)
     + 160;
  v5 = sub_22077B0(0x40u);
  v6 = (__int64 *)v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = v4;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    *(_DWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    sub_39524F0((unsigned int *)v5);
    while ( (unsigned __int8)sub_3951AF0(v6) )
      ;
  }
  v7 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v6;
  if ( v7 )
  {
    v8 = *(_QWORD *)(v7 + 40);
    if ( v8 )
      j_j___libc_free_0(v8);
    v9 = *(unsigned int *)(v7 + 32);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD **)(v7 + 16);
      v11 = &v10[5 * v9];
      do
      {
        if ( *v10 != -8 && *v10 != -16 )
          _libc_free(v10[2]);
        v10 += 5;
      }
      while ( v11 != v10 );
    }
    j___libc_free_0(*(_QWORD *)(v7 + 16));
    j_j___libc_free_0(v7);
  }
  return 0;
}
