// Function: sub_2427A00
// Address: 0x2427a00
//
__int64 __fastcall sub_2427A00(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        unsigned __int64 *a4,
        __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r9
  unsigned __int64 v12; // rax
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // r15
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rax
  bool v19; // zf
  unsigned __int64 v20; // rdi
  __int64 v21; // r15
  __int64 v22; // r13
  __int64 v23; // rbx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __int64 v26; // [rsp-40h] [rbp-40h]

  if ( a3 == 1 )
    return (__int64)a1;
  if ( a3 <= a5 )
  {
    v12 = *a1;
    v13 = a1;
    *a1 = 0;
    v14 = a4 + 1;
    v15 = *a4;
    v16 = v13 + 1;
    *a4 = v12;
    if ( v15 )
    {
      j_j___libc_free_0(v15);
      if ( a2 != v16 )
        goto LABEL_15;
    }
    else if ( a2 != v16 )
    {
      do
      {
LABEL_15:
        while ( 1 )
        {
          v18 = *v16;
          v19 = *(_QWORD *)(*v16 + 24) == 0;
          *v16 = 0;
          if ( !v19 )
            break;
          v20 = *v14;
          *v14 = v18;
          if ( v20 )
            j_j___libc_free_0(v20);
          ++v16;
          ++v14;
          if ( a2 == v16 )
            goto LABEL_19;
        }
        v17 = *v13;
        *v13 = v18;
        if ( v17 )
          j_j___libc_free_0(v17);
        ++v16;
        ++v13;
      }
      while ( a2 != v16 );
LABEL_19:
      v21 = (char *)v14 - (char *)a4;
      v22 = v21 >> 3;
      if ( v21 <= 0 )
        return (__int64)v13;
      goto LABEL_20;
    }
    v22 = 1;
LABEL_20:
    v23 = 0;
    do
    {
      v24 = a4[v23];
      a4[v23] = 0;
      v25 = v13[v23];
      v13[v23] = v24;
      if ( v25 )
        j_j___libc_free_0(v25);
      ++v23;
    }
    while ( v22 - v23 > 0 );
    return (__int64)v13;
  }
  v7 = (__int64)&a1[a3 / 2];
  v26 = a3 / 2;
  v8 = sub_2427A00(a1, v7, a3 / 2);
  v9 = a3 - v26;
  v10 = v7;
  if ( v9 )
  {
    while ( *(_QWORD *)(*(_QWORD *)v10 + 24LL) )
    {
      v10 += 8;
      if ( !--v9 )
        return sub_24252C0(v8, v7, v10);
    }
    v10 = sub_2427A00(v10, a2, v9);
  }
  return sub_24252C0(v8, v7, v10);
}
