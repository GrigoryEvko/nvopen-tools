// Function: sub_1E2C960
// Address: 0x1e2c960
//
__int64 __fastcall sub_1E2C960(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r14
  _QWORD *v5; // r12
  _QWORD *v6; // r14
  __int64 v7; // rdi
  _QWORD *v8; // r14
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // r12
  _QWORD *v13; // r14
  __int64 v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 *v16; // r15
  __int64 v17; // r12
  __int64 (__fastcall *v18)(__int64); // rax

  v2 = a1[213];
  if ( v2 != a1[214] )
    a1[214] = v2;
  v3 = a1[217];
  if ( v3 )
  {
    v4 = *(unsigned int *)(v3 + 88);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v3 + 72);
      v6 = &v5[4 * v4];
      do
      {
        if ( *v5 != -16 && *v5 != -8 )
        {
          v7 = v5[1];
          if ( v7 )
            j_j___libc_free_0(v7, v5[3] - v7);
        }
        v5 += 4;
      }
      while ( v6 != v5 );
    }
    j___libc_free_0(*(_QWORD *)(v3 + 72));
    v8 = *(_QWORD **)(v3 + 48);
    v9 = *(_QWORD **)(v3 + 40);
    if ( v8 != v9 )
    {
      do
      {
        v10 = v9[3];
        *v9 = &unk_49EE2B0;
        if ( v10 != 0 && v10 != -8 && v10 != -16 )
          sub_1649B30(v9 + 1);
        v9 += 5;
      }
      while ( v8 != v9 );
      v9 = *(_QWORD **)(v3 + 40);
    }
    if ( v9 )
      j_j___libc_free_0(v9, *(_QWORD *)(v3 + 56) - (_QWORD)v9);
    v11 = *(unsigned int *)(v3 + 32);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD **)(v3 + 16);
      v13 = &v12[4 * v11];
      do
      {
        if ( *v12 != -8 && *v12 != -16 )
        {
          v14 = v12[1];
          if ( (v14 & 4) != 0 )
          {
            v15 = (unsigned __int64 *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
            v16 = v15;
            if ( v15 )
            {
              if ( (unsigned __int64 *)*v15 != v15 + 2 )
                _libc_free(*v15);
              j_j___libc_free_0(v16, 48);
            }
          }
        }
        v12 += 4;
      }
      while ( v13 != v12 );
    }
    j___libc_free_0(*(_QWORD *)(v3 + 16));
    j_j___libc_free_0(v3, 96);
  }
  a1[217] = 0;
  sub_38C0300(a1 + 21);
  v17 = a1[212];
  if ( v17 )
  {
    v18 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL);
    if ( v18 == sub_1E2C780 )
    {
      nullsub_703();
      j_j___libc_free_0(v17, 8);
    }
    else
    {
      v18(a1[212]);
    }
  }
  a1[212] = 0;
  return 0;
}
