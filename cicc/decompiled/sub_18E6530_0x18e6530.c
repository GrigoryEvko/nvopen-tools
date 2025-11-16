// Function: sub_18E6530
// Address: 0x18e6530
//
void *__fastcall sub_18E6530(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r14
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12

  v1 = *(unsigned int *)(a1 + 304);
  v2 = *(_QWORD *)(a1 + 296);
  *(_QWORD *)a1 = off_49F2D08;
  v3 = v2 + 632 * v1;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(unsigned int *)(v3 - 616);
      v5 = *(_QWORD *)(v3 - 624);
      v3 -= 632LL;
      v6 = (unsigned __int64 *)(v5 + 152 * v4);
      if ( (unsigned __int64 *)v5 != v6 )
      {
        do
        {
          v6 -= 19;
          if ( (unsigned __int64 *)*v6 != v6 + 2 )
            _libc_free(*v6);
        }
        while ( (unsigned __int64 *)v5 != v6 );
        v5 = *(_QWORD *)(v3 + 8);
      }
      if ( v5 != v3 + 24 )
        _libc_free(v5);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 296);
  }
  if ( v3 != a1 + 312 )
    _libc_free(v3);
  if ( (*(_BYTE *)(a1 + 224) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 232));
  v7 = *(unsigned __int64 **)(a1 + 200);
  v8 = *(unsigned __int64 **)(a1 + 192);
  if ( v7 != v8 )
  {
    do
    {
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        _libc_free(*v8);
      v8 += 20;
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 192);
  }
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 208) - (_QWORD)v8);
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
