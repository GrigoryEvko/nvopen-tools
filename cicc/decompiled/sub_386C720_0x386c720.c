// Function: sub_386C720
// Address: 0x386c720
//
__int64 __fastcall sub_386C720(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 *v21; // rax
  __int64 *v22; // r8
  __int64 *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx

  if ( a1[79] )
  {
    v21 = (__int64 *)a1[76];
    v22 = a1 + 75;
    if ( v21 )
    {
      v23 = a1 + 75;
      do
      {
        while ( 1 )
        {
          v24 = v21[2];
          v25 = v21[3];
          if ( a2 <= v21[4] )
            break;
          v21 = (__int64 *)v21[3];
          if ( !v25 )
            goto LABEL_23;
        }
        v23 = v21;
        v21 = (__int64 *)v21[2];
      }
      while ( v24 );
LABEL_23:
      if ( v22 != v23 && a2 >= v23[4] )
        return a2;
    }
  }
  else
  {
    v14 = (_QWORD *)a1[64];
    v15 = &v14[*((unsigned int *)a1 + 130)];
    if ( v14 != v15 )
    {
      while ( a2 != *v14 )
      {
        if ( v15 == ++v14 )
          goto LABEL_3;
      }
      if ( v15 != v14 )
        return a2;
    }
  }
LABEL_3:
  v16 = *a3;
  v17 = *a3 + 24LL * *((unsigned int *)a3 + 2);
  if ( *a3 == v17 )
    return *(_QWORD *)(*a1 + 120);
  v18 = 0;
  do
  {
    v19 = *(_QWORD *)(v16 + 16);
    if ( a2 != v19 && v19 != v18 )
    {
      if ( v18 )
        return a2;
      v18 = *(_QWORD *)(v16 + 16);
    }
    v16 += 24;
  }
  while ( v17 != v16 );
  if ( !v18 )
    return *(_QWORD *)(*a1 + 120);
  if ( a2 )
  {
    sub_164D160(a2, v18, a4, a5, a6, a7, a8, a9, a10, a11);
    sub_386B550(a1, a2);
  }
  return sub_386C330((__int64)a1, v18, a4, a5, a6, a7, a8, a9, a10, a11);
}
