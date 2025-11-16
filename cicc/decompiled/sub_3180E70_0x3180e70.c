// Function: sub_3180E70
// Address: 0x3180e70
//
__int64 __fastcall sub_3180E70(unsigned __int64 *a1, int *a2, size_t a3, char a4)
{
  size_t v4; // r14
  __int64 v7; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 **v10; // r8
  unsigned __int64 *v11; // rax
  _QWORD *v12; // rsi
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rbx
  unsigned __int64 i; // r14
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r9
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // r8
  unsigned __int64 *v22; // rax
  size_t v23; // [rsp+8h] [rbp-108h] BYREF
  unsigned __int64 *v24[2]; // [rsp+10h] [rbp-100h] BYREF
  _QWORD v25[4]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 *v26[26]; // [rsp+40h] [rbp-D0h] BYREF

  v4 = a3;
  v7 = (__int64)sub_317EE00((__int64)a1, a2, a3);
  if ( !a4 )
    goto LABEL_2;
  memset(v25, 0, 24);
  if ( a2 )
  {
    sub_C7D030(v26);
    sub_C7D280((int *)v26, a2, v4);
    sub_C7D290(v26, v24);
    v4 = (size_t)v24[0];
  }
  v9 = a1[1];
  v23 = v4;
  v10 = *(unsigned __int64 ***)(*a1 + 8 * (v4 % v9));
  if ( v10 )
  {
    v11 = *v10;
    if ( (*v10)[1] == v4 )
    {
LABEL_11:
      v13 = *v10;
      if ( *v10 )
      {
        v14 = v25[0];
        if ( !v25[0] )
          goto LABEL_14;
        goto LABEL_13;
      }
    }
    else
    {
      while ( 1 )
      {
        v12 = (_QWORD *)*v11;
        if ( !*v11 )
          break;
        v10 = (unsigned __int64 **)v11;
        if ( v4 % v9 != v12[1] % v9 )
          break;
        v11 = (unsigned __int64 *)*v11;
        if ( v12[1] == v4 )
          goto LABEL_11;
      }
    }
  }
  v26[0] = v25;
  v24[0] = &v23;
  v22 = sub_317D460(a1, v24, v26);
  v14 = v25[0];
  v13 = v22;
  if ( v25[0] )
LABEL_13:
    j_j___libc_free_0(v14);
LABEL_14:
  v15 = v13[2];
  for ( i = v13[3]; i != v15; v15 += 8LL )
  {
    v17 = *(_QWORD *)v15;
    if ( (*(_BYTE *)(*(_QWORD *)v15 + 48LL) & 0xC) == 0 )
    {
      v18 = a1[8];
      v19 = *(_QWORD **)(a1[7] + 8 * (v17 % v18));
      if ( v19 )
      {
        v20 = (_QWORD *)*v19;
        if ( v17 == *(_QWORD *)(*v19 + 8LL) )
        {
LABEL_23:
          v19 = (_QWORD *)*v19;
          if ( v19 )
            v19 = (_QWORD *)v19[2];
        }
        else
        {
          while ( 1 )
          {
            v21 = *v20;
            if ( !*v20 )
              break;
            v19 = v20;
            if ( v17 % v18 != *(_QWORD *)(v21 + 8) % v18 )
              break;
            v20 = (_QWORD *)*v20;
            if ( v17 == *(_QWORD *)(v21 + 8) )
              goto LABEL_23;
          }
          v19 = 0;
        }
      }
      if ( v19 != (_QWORD *)v7 )
        v7 = sub_3180E40((__int64)a1, (__int64)v19);
    }
  }
LABEL_2:
  if ( v7 )
    return sub_317E470(v7);
  else
    return 0;
}
