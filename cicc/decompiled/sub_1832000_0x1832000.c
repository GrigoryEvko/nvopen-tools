// Function: sub_1832000
// Address: 0x1832000
//
__int64 __fastcall sub_1832000(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // r14
  __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 *v8; // r15

  *(_QWORD *)a1 = &unk_49F1848;
  v2 = *(_QWORD *)(a1 + 208);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 224) - v2);
  v3 = *(_QWORD *)(a1 + 176);
  if ( *(_DWORD *)(a1 + 188) )
  {
    v4 = *(unsigned int *)(a1 + 184);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD *)(v3 + v6);
        if ( v7 != -8 && v7 )
        {
          v8 = *(unsigned __int64 **)(v7 + 8);
          if ( v8 )
          {
            if ( (unsigned __int64 *)*v8 != v8 + 2 )
              _libc_free(*v8);
            j_j___libc_free_0(v8, 96);
          }
          _libc_free(v7);
          v3 = *(_QWORD *)(a1 + 176);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3);
  *(_QWORD *)a1 = &unk_4A3DD58;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 280);
}
