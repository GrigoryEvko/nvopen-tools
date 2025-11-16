// Function: sub_1F603E0
// Address: 0x1f603e0
//
void *__fastcall sub_1F603E0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // rbx
  __int64 v8; // rax
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r14

  v2 = *(_QWORD *)(a1 + 240);
  v3 = *(_QWORD *)(a1 + 232);
  *(_QWORD *)a1 = off_49FFDC0;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 )
        j_j___libc_free_0(v4, *(_QWORD *)(v3 + 24) - v4);
      v3 += 32;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 232);
  }
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 248) - v3);
  j___libc_free_0(*(_QWORD *)(a1 + 208));
  v5 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 176);
    v7 = &v6[2 * v5];
    do
    {
      if ( *v6 != -16 && *v6 != -8 )
      {
        v8 = v6[1];
        if ( (v8 & 4) != 0 )
        {
          v9 = (unsigned __int64 *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
          v10 = v9;
          if ( v9 )
          {
            if ( (unsigned __int64 *)*v9 != v9 + 2 )
              _libc_free(*v9);
            j_j___libc_free_0(v10, 48);
          }
        }
      }
      v6 += 2;
    }
    while ( v7 != v6 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 176));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
