// Function: sub_1DAA110
// Address: 0x1daa110
//
void *__fastcall sub_1DAA110(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 *v12; // r12
  __int64 v13; // rax
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // r12
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  _DWORD *v21; // rax

  v2 = a1[29];
  *a1 = off_49FAC40;
  if ( v2 )
  {
    j___libc_free_0(*(_QWORD *)(v2 + 272));
    j___libc_free_0(*(_QWORD *)(v2 + 240));
    v3 = *(_QWORD *)(v2 + 152);
    v4 = v3 + 8LL * *(unsigned int *)(v2 + 160);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 - 8);
        v4 -= 8LL;
        if ( v5 )
        {
          sub_1DA8360(*(_QWORD *)(v5 + 360));
          v9 = *(_QWORD *)(v5 + 312);
          if ( v9 != v5 + 328 )
            _libc_free(v9);
          if ( *(_DWORD *)(v5 + 296) )
          {
            sub_1DA9BF0(v5 + 216, (char *)sub_1DA8010, 0, v6, v7, v8);
            v21 = (_DWORD *)(v5 + 280);
            do
              *v21++ = 0;
            while ( (_DWORD *)(v5 + 296) != v21 );
          }
          v10 = *(_QWORD *)(v5 + 40);
          if ( v10 != v5 + 56 )
            _libc_free(v10);
          v11 = *(_QWORD *)(v5 + 16);
          if ( v11 )
            sub_161E7C0(v5 + 16, v11);
          j_j___libc_free_0(v5, 392);
        }
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v2 + 152);
    }
    if ( v4 != v2 + 168 )
      _libc_free(v4);
    v12 = *(unsigned __int64 **)(v2 + 32);
    v13 = *(unsigned int *)(v2 + 40);
    *(_QWORD *)(v2 + 8) = 0;
    v14 = &v12[v13];
    while ( v14 != v12 )
    {
      v15 = *v12++;
      _libc_free(v15);
    }
    v16 = *(unsigned __int64 **)(v2 + 80);
    v17 = (unsigned __int64)&v16[2 * *(unsigned int *)(v2 + 88)];
    if ( v16 != (unsigned __int64 *)v17 )
    {
      do
      {
        v18 = *v16;
        v16 += 2;
        _libc_free(v18);
      }
      while ( (unsigned __int64 *)v17 != v16 );
      v17 = *(_QWORD *)(v2 + 80);
    }
    if ( v17 != v2 + 96 )
      _libc_free(v17);
    v19 = *(_QWORD *)(v2 + 32);
    if ( v19 != v2 + 48 )
      _libc_free(v19);
    j_j___libc_free_0(v2, 296);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
