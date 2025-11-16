// Function: sub_1E2C570
// Address: 0x1e2c570
//
__int64 __fastcall sub_1E2C570(_QWORD *a1)
{
  __int64 v1; // r12
  void (*v2)(void); // rax
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // r13
  _QWORD *v12; // r14
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // [rsp+8h] [rbp-38h]

  v1 = (__int64)(a1 + 29);
  v2 = *(void (**)(void))(*a1 + 96LL);
  if ( (char *)v2 == (char *)sub_1E290A0 )
    sub_1E28CF0((__int64)(a1 + 29));
  else
    v2();
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
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
    v8 = sub_22077B0(80);
    if ( v8 )
    {
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)v8 = v8 + 16;
      *(_QWORD *)(v8 + 8) = 0x100000000LL;
      *(_QWORD *)(v8 + 32) = 0;
      *(_QWORD *)(v8 + 40) = 0;
      *(_DWORD *)(v8 + 48) = 0;
      *(_QWORD *)(v8 + 64) = 0;
      *(_BYTE *)(v8 + 72) = 0;
      *(_DWORD *)(v8 + 76) = 0;
    }
    v9 = *(_QWORD *)(v6 + 1312);
    *(_QWORD *)(v6 + 1312) = v8;
    if ( v9 )
    {
      v10 = *(unsigned int *)(v9 + 48);
      if ( (_DWORD)v10 )
      {
        v11 = *(_QWORD **)(v9 + 32);
        v12 = &v11[2 * v10];
        do
        {
          if ( *v11 != -8 && *v11 != -16 )
          {
            v13 = v11[1];
            if ( v13 )
            {
              v14 = *(_QWORD *)(v13 + 24);
              if ( v14 )
              {
                v15 = v11[1];
                j_j___libc_free_0(v14, *(_QWORD *)(v13 + 40) - v14);
                v13 = v15;
              }
              j_j___libc_free_0(v13, 56);
            }
          }
          v11 += 2;
        }
        while ( v12 != v11 );
      }
      j___libc_free_0(*(_QWORD *)(v9 + 32));
      if ( *(_QWORD *)v9 != v9 + 16 )
        _libc_free(*(_QWORD *)v9);
      j_j___libc_free_0(v9, 80);
    }
  }
  sub_1E06620(v6);
  sub_1E2B150(v1, *(_QWORD *)(v6 + 1312));
  return 0;
}
