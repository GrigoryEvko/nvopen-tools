// Function: sub_1BFBE90
// Address: 0x1bfbe90
//
__int64 __fastcall sub_1BFBE90(__int64 a1)
{
  int v2; // eax
  _QWORD *v3; // rdi
  _QWORD *v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  unsigned __int64 **v8; // r14
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // r15
  unsigned __int64 *v11; // r15
  unsigned __int64 *v12; // r15

  *(_QWORD *)a1 = &unk_49F72A0;
  v2 = *(_DWORD *)(a1 + 256);
  v3 = *(_QWORD **)(a1 + 248);
  if ( v2 )
  {
    v5 = &v3[2 * *(unsigned int *)(a1 + 264)];
    if ( v5 != v3 )
    {
      v6 = v3;
      while ( 1 )
      {
        v7 = v6;
        if ( *v6 != -16 && *v6 != -8 )
          break;
        v6 += 2;
        if ( v5 == v6 )
          goto LABEL_2;
      }
      if ( v5 != v6 )
      {
        do
        {
          v8 = (unsigned __int64 **)v7[1];
          if ( v8 )
          {
            v9 = *v8;
            if ( *v8 )
            {
              _libc_free(*v9);
              j_j___libc_free_0(v9, 24);
            }
            v10 = v8[1];
            if ( v10 )
            {
              _libc_free(*v10);
              j_j___libc_free_0(v10, 24);
            }
            v11 = v8[2];
            if ( v11 )
            {
              _libc_free(*v11);
              j_j___libc_free_0(v11, 24);
            }
            v12 = v8[3];
            if ( v12 )
            {
              _libc_free(*v12);
              j_j___libc_free_0(v12, 24);
            }
            j_j___libc_free_0(v8, 32);
          }
          v7 += 2;
          if ( v7 == v5 )
            break;
          while ( *v7 == -8 || *v7 == -16 )
          {
            v7 += 2;
            if ( v5 == v7 )
              goto LABEL_23;
          }
        }
        while ( v5 != v7 );
LABEL_23:
        v3 = *(_QWORD **)(a1 + 248);
      }
    }
  }
LABEL_2:
  j___libc_free_0(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 208));
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 288);
}
