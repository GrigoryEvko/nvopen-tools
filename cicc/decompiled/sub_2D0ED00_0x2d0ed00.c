// Function: sub_2D0ED00
// Address: 0x2d0ed00
//
unsigned __int64 __fastcall sub_2D0ED00(
        unsigned __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5)
{
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  char *v11; // r15
  _QWORD *v12; // rax
  unsigned __int64 v14; // r13
  size_t v15; // r12
  unsigned __int64 *v16; // r9
  _QWORD *v17; // rsi
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rcx
  unsigned __int64 v20; // rdx
  char *v21; // rax

  v8 = (__int64)(a1 + 4);
  v9 = *(_QWORD *)(v8 - 24);
  if ( !(unsigned __int8)sub_222DA10(v8, v9, *(_QWORD *)(v8 - 8), a5) )
  {
    v11 = (char *)*a1;
    goto LABEL_3;
  }
  v14 = v10;
  if ( v10 == 1 )
  {
    v11 = (char *)(a1 + 6);
    a1[6] = 0;
    v16 = a1 + 6;
  }
  else
  {
    if ( v10 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(v8, v9, v10);
    v15 = 8 * v10;
    v11 = (char *)sub_22077B0(8 * v10);
    memset(v11, 0, v15);
    v16 = a1 + 6;
  }
  v17 = (_QWORD *)a1[2];
  a1[2] = 0;
  if ( v17 )
  {
    v18 = 0;
    do
    {
      while ( 1 )
      {
        v19 = v17;
        v17 = (_QWORD *)*v17;
        v20 = v19[1] % v14;
        v21 = &v11[8 * v20];
        if ( !*(_QWORD *)v21 )
          break;
        *v19 = **(_QWORD **)v21;
        **(_QWORD **)v21 = v19;
LABEL_12:
        if ( !v17 )
          goto LABEL_16;
      }
      *v19 = a1[2];
      a1[2] = (unsigned __int64)v19;
      *(_QWORD *)v21 = a1 + 2;
      if ( !*v19 )
      {
        v18 = v20;
        goto LABEL_12;
      }
      *(_QWORD *)&v11[8 * v18] = v19;
      v18 = v20;
    }
    while ( v17 );
  }
LABEL_16:
  if ( (unsigned __int64 *)*a1 != v16 )
    j_j___libc_free_0(*a1);
  a1[1] = v14;
  *a1 = (unsigned __int64)v11;
  a2 = a3 % v14;
LABEL_3:
  v12 = *(_QWORD **)&v11[8 * a2];
  if ( v12 )
  {
    *(_QWORD *)a4 = *v12;
    **(_QWORD **)(*a1 + 8 * a2) = a4;
  }
  else
  {
    *(_QWORD *)a4 = a1[2];
    a1[2] = a4;
    if ( *(_QWORD *)a4 )
      *(_QWORD *)(*a1 + 8 * (*(_QWORD *)(*(_QWORD *)a4 + 8LL) % a1[1])) = a4;
    *(_QWORD *)(*a1 + 8 * a2) = a1 + 2;
  }
  ++a1[3];
  return a4;
}
