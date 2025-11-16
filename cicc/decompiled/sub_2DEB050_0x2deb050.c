// Function: sub_2DEB050
// Address: 0x2deb050
//
void __fastcall sub_2DEB050(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // r12
  unsigned __int64 v10; // rdx
  int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // rbx
  unsigned int v17; // eax
  const void **v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rbx
  bool v28; // cc
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rdi
  __int64 v32; // rcx
  unsigned __int64 v33; // [rsp-48h] [rbp-48h]
  __int64 v34; // [rsp-40h] [rbp-40h]
  __int64 v35; // [rsp-40h] [rbp-40h]
  unsigned __int64 v36; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *((unsigned int *)a2 + 2);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = v8;
    v12 = *(_QWORD *)a1;
    if ( v8 <= v10 )
    {
      v20 = *(_QWORD *)a1;
      if ( v8 )
      {
        v25 = *a2;
        v26 = v9 + 8;
        v27 = v25 + 8;
        v35 = v25 + 8 + 24 * v8;
        do
        {
          v28 = *(_DWORD *)(v26 + 8) <= 0x40u;
          *(_DWORD *)(v26 - 8) = *(_DWORD *)(v27 - 8);
          if ( v28 && *(_DWORD *)(v27 + 8) <= 0x40u )
          {
            *(_QWORD *)v26 = *(_QWORD *)v27;
            *(_DWORD *)(v26 + 8) = *(_DWORD *)(v27 + 8);
          }
          else
          {
            sub_C43990(v26, v27);
          }
          v26 += 24;
          v27 += 24;
        }
        while ( v35 != v27 );
        v20 = *(_QWORD *)a1;
        v10 = *(unsigned int *)(a1 + 8);
        v12 = v9 + 24 * v8;
      }
      v21 = v20 + 24 * v10;
      while ( v12 != v21 )
      {
        v21 -= 24;
        if ( *(_DWORD *)(v21 + 16) > 0x40u )
        {
          v22 = *(_QWORD *)(v21 + 8);
          if ( v22 )
          {
            v34 = v12;
            j_j___libc_free_0_0(v22);
            v12 = v34;
          }
        }
      }
LABEL_12:
      *(_DWORD *)(a1 + 8) = v11;
      return;
    }
    if ( v8 > *(unsigned int *)(a1 + 12) )
    {
      v23 = v9 + 24 * v10;
      while ( v23 != v9 )
      {
        while ( 1 )
        {
          v23 -= 24;
          if ( *(_DWORD *)(v23 + 16) <= 0x40u )
            break;
          v24 = *(_QWORD *)(v23 + 8);
          if ( !v24 )
            break;
          j_j___libc_free_0_0(v24);
          if ( v23 == v9 )
            goto LABEL_25;
        }
      }
LABEL_25:
      *(_DWORD *)(a1 + 8) = 0;
      sub_2DEABD0(a1, v8, v10, a4, a5, a6);
      v8 = *((unsigned int *)a2 + 2);
      v9 = *(_QWORD *)a1;
      v10 = 0;
    }
    else if ( *(_DWORD *)(a1 + 8) )
    {
      v10 *= 24LL;
      v29 = v9 + 8;
      v30 = *a2 + 8;
      v33 = v30 + v10;
      do
      {
        while ( 1 )
        {
          v28 = *(_DWORD *)(v29 + 8) <= 0x40u;
          *(_DWORD *)(v29 - 8) = *(_DWORD *)(v30 - 8);
          if ( v28 && *(_DWORD *)(v30 + 8) <= 0x40u )
            break;
          v31 = v29;
          v36 = v10;
          v29 += 24;
          sub_C43990(v31, v30);
          v30 += 24;
          v10 = v36;
          if ( v33 == v30 )
            goto LABEL_38;
        }
        v32 = *(_QWORD *)v30;
        v29 += 24;
        v30 += 24;
        *(_QWORD *)(v29 - 24) = v32;
        *(_DWORD *)(v29 - 16) = *(_DWORD *)(v30 - 16);
      }
      while ( v33 != v30 );
LABEL_38:
      v8 = *((unsigned int *)a2 + 2);
      v9 = *(_QWORD *)a1;
    }
    v13 = *a2;
    v14 = v10 + v9;
    v15 = *a2 + 24 * v8;
    v16 = v10 + v13;
    if ( v15 == v16 )
      goto LABEL_12;
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_8;
      *(_DWORD *)v14 = *(_DWORD *)v16;
      v17 = *(_DWORD *)(v16 + 16);
      *(_DWORD *)(v14 + 16) = v17;
      if ( v17 <= 0x40 )
      {
        *(_QWORD *)(v14 + 8) = *(_QWORD *)(v16 + 8);
LABEL_8:
        v16 += 24;
        v14 += 24;
        if ( v15 == v16 )
          goto LABEL_12;
      }
      else
      {
        v18 = (const void **)(v16 + 8);
        v19 = v14 + 8;
        v16 += 24;
        v14 += 24;
        sub_C43780(v19, v18);
        if ( v15 == v16 )
          goto LABEL_12;
      }
    }
  }
}
