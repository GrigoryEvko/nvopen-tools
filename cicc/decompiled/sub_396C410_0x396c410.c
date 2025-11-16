// Function: sub_396C410
// Address: 0x396c410
//
void *__fastcall sub_396C410(_QWORD *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // r14
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // r12
  __int64 *v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rdi

  v2 = a1[51];
  *a1 = &unk_4A3F580;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 24);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 8);
      v5 = &v4[2 * v3];
      do
      {
        if ( *v4 != -16 && *v4 != -8 )
        {
          v6 = v4[1];
          if ( v6 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
        }
        v4 += 2;
      }
      while ( v5 != v4 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 8));
    j_j___libc_free_0(v2);
    a1[51] = 0;
  }
  v7 = a1[68];
  if ( (_QWORD *)v7 != a1 + 70 )
    _libc_free(v7);
  v8 = (unsigned __int64 *)a1[62];
  if ( v8 )
  {
    v9 = v8[8];
    if ( v9 )
      j_j___libc_free_0(v9);
    v10 = (unsigned __int64 *)v8[4];
    v11 = (unsigned __int64 *)v8[3];
    if ( v10 != v11 )
    {
      do
      {
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11);
        v11 += 4;
      }
      while ( v10 != v11 );
      v11 = (unsigned __int64 *)v8[3];
    }
    if ( v11 )
      j_j___libc_free_0((unsigned __int64)v11);
    v12 = v8[1];
    v13 = *v8;
    if ( v12 != *v8 )
    {
      do
      {
        v14 = (__int64 *)v13;
        v13 += 24LL;
        sub_16CE300(v14);
      }
      while ( v12 != v13 );
      v13 = *v8;
    }
    if ( v13 )
      j_j___libc_free_0(v13);
    j_j___libc_free_0((unsigned __int64)v8);
  }
  v15 = a1[61];
  if ( v15 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
  v16 = a1[60];
  if ( v16 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
  v17 = a1[53];
  if ( (_QWORD *)v17 != a1 + 55 )
    _libc_free(v17);
  v18 = a1[44];
  if ( v18 )
    j_j___libc_free_0(v18);
  j___libc_free_0(a1[41]);
  v19 = a1[32];
  if ( v19 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
