// Function: sub_21070E0
// Address: 0x21070e0
//
__int64 __fastcall sub_21070E0(__int64 a1)
{
  void (*v1)(void); // rax
  __int64 *v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // r13
  _QWORD *v16; // r14
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-38h]

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 96LL);
  if ( (char *)v1 == (char *)sub_2105CF0 )
  {
    sub_2105B40(*(_QWORD **)(a1 + 248));
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 256) = a1 + 240;
    *(_QWORD *)(a1 + 264) = a1 + 240;
    *(_QWORD *)(a1 + 272) = 0;
  }
  else
  {
    v1();
  }
  v2 = *(__int64 **)(a1 + 8);
  v3 = a1 + 232;
  v4 = *v2;
  v5 = v2[1];
  if ( v4 == v5 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4FC62EC )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_25;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4FC62EC);
  if ( !*(_QWORD *)(v6 + 1312) )
  {
    v12 = sub_22077B0(80);
    if ( v12 )
    {
      *(_QWORD *)(v12 + 24) = 0;
      *(_QWORD *)v12 = v12 + 16;
      *(_QWORD *)(v12 + 8) = 0x100000000LL;
      *(_QWORD *)(v12 + 32) = 0;
      *(_QWORD *)(v12 + 40) = 0;
      *(_DWORD *)(v12 + 48) = 0;
      *(_QWORD *)(v12 + 64) = 0;
      *(_BYTE *)(v12 + 72) = 0;
      *(_DWORD *)(v12 + 76) = 0;
    }
    v13 = *(_QWORD *)(v6 + 1312);
    *(_QWORD *)(v6 + 1312) = v12;
    if ( v13 )
    {
      v14 = *(unsigned int *)(v13 + 48);
      if ( (_DWORD)v14 )
      {
        v15 = *(_QWORD **)(v13 + 32);
        v16 = &v15[2 * v14];
        do
        {
          if ( *v15 != -8 && *v15 != -16 )
          {
            v17 = v15[1];
            if ( v17 )
            {
              v18 = *(_QWORD *)(v17 + 24);
              if ( v18 )
              {
                v19 = v15[1];
                j_j___libc_free_0(v18, *(_QWORD *)(v17 + 40) - v18);
                v17 = v19;
              }
              j_j___libc_free_0(v17, 56);
            }
          }
          v15 += 2;
        }
        while ( v16 != v15 );
      }
      j___libc_free_0(*(_QWORD *)(v13 + 32));
      if ( *(_QWORD *)v13 != v13 + 16 )
        _libc_free(*(_QWORD *)v13);
      j_j___libc_free_0(v13, 80);
    }
  }
  sub_1E06620(v6);
  sub_2106FE0(v3, *(_QWORD *)(v6 + 1312), v7, v8, v9, v10);
  return 0;
}
