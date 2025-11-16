// Function: sub_2C72470
// Address: 0x2c72470
//
__int64 __fastcall sub_2C72470(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  unsigned __int64 v15; // rdi

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_21:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F8144C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_21;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F8144C)
     + 176;
  v5 = sub_22077B0(0x40u);
  v9 = (__int64 *)v5;
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
    sub_2C71E10((_QWORD *)v5, (__int64)&unk_4F8144C, v6, v7, v8);
    while ( (unsigned __int8)sub_2C70040(v9) )
      ;
  }
  v10 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v9;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 40);
    if ( v11 )
      j_j___libc_free_0(v11);
    v12 = *(unsigned int *)(v10 + 32);
    if ( (_DWORD)v12 )
    {
      v13 = *(_QWORD **)(v10 + 16);
      v14 = &v13[11 * v12];
      do
      {
        if ( *v13 != -4096 && *v13 != -8192 )
        {
          v15 = v13[2];
          if ( (_QWORD *)v15 != v13 + 4 )
            _libc_free(v15);
        }
        v13 += 11;
      }
      while ( v14 != v13 );
      v12 = *(unsigned int *)(v10 + 32);
    }
    sub_C7D6A0(*(_QWORD *)(v10 + 16), 88 * v12, 8);
    j_j___libc_free_0(v10);
  }
  return 0;
}
