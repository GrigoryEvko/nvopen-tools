// Function: sub_154BC70
// Address: 0x154bc70
//
__int64 __fastcall sub_154BC70(__int64 a1)
{
  __int64 v3; // r13
  char v4; // r14
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  unsigned __int64 v9; // r8
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int64 v12; // rdi

  if ( !*(_BYTE *)(a1 + 8) )
    return *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_BYTE *)(a1 + 9);
  *(_BYTE *)(a1 + 8) = 0;
  v5 = sub_22077B0(272);
  v6 = v5;
  if ( v5 )
    sub_154BB30(v5, v3, v4);
  v7 = *(_QWORD *)a1;
  *(_QWORD *)a1 = v6;
  if ( v7 )
  {
    j___libc_free_0(*(_QWORD *)(v7 + 240));
    if ( *(_DWORD *)(v7 + 204) )
    {
      v8 = *(unsigned int *)(v7 + 200);
      v9 = *(_QWORD *)(v7 + 192);
      if ( (_DWORD)v8 )
      {
        v10 = 8 * v8;
        v11 = 0;
        do
        {
          v12 = *(_QWORD *)(v9 + v11);
          if ( v12 != -8 )
          {
            if ( v12 )
            {
              _libc_free(v12);
              v9 = *(_QWORD *)(v7 + 192);
            }
          }
          v11 += 8;
        }
        while ( v10 != v11 );
      }
    }
    else
    {
      v9 = *(_QWORD *)(v7 + 192);
    }
    _libc_free(v9);
    j___libc_free_0(*(_QWORD *)(v7 + 160));
    j___libc_free_0(*(_QWORD *)(v7 + 120));
    j___libc_free_0(*(_QWORD *)(v7 + 80));
    j___libc_free_0(*(_QWORD *)(v7 + 40));
    j_j___libc_free_0(v7, 272);
    v6 = *(_QWORD *)a1;
  }
  *(_QWORD *)(a1 + 32) = v6;
  return v6;
}
