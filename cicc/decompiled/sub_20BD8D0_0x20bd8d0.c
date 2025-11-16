// Function: sub_20BD8D0
// Address: 0x20bd8d0
//
void __fastcall sub_20BD8D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r8
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdx
  _QWORD *v10; // r15
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  __int64 *v14; // rbx
  __int64 v15; // r13
  unsigned __int64 i; // r12
  __int64 v17; // r15
  _QWORD *v18; // rbx
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  _QWORD *v21; // r12
  __int64 v22; // rbx
  _QWORD *v23; // r12
  __int64 v24; // rsi
  _QWORD *v25; // rdi
  unsigned int v26; // [rsp-44h] [rbp-44h]
  unsigned __int64 v27; // [rsp-40h] [rbp-40h]
  unsigned __int64 v28; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *(_QWORD **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v26 = *((_DWORD *)a2 + 2);
    v7 = v26;
    v10 = *(_QWORD **)a1;
    if ( v26 <= v9 )
    {
      v20 = *(_QWORD **)a1;
      if ( v26 )
      {
        v22 = *a2;
        v23 = &v8[4 * v26];
        do
        {
          v24 = v22;
          v25 = v8;
          v8 += 4;
          v22 += 32;
          sub_2240AE0(v25, v24);
        }
        while ( v8 != v23 );
        v20 = *(_QWORD **)a1;
        v9 = *(unsigned int *)(a1 + 8);
      }
      v21 = &v20[4 * v9];
      while ( v8 != v21 )
      {
        v21 -= 4;
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
    }
    else
    {
      v11 = *(unsigned int *)(a1 + 12);
      if ( v26 <= v11 )
      {
        if ( *(_DWORD *)(a1 + 8) )
        {
          v9 *= 32LL;
          v17 = *a2;
          v18 = (_QWORD *)((char *)v8 + v9);
          do
          {
            v19 = v8;
            v8 += 4;
            v28 = v9;
            sub_2240AE0(v19, v17);
            v17 += 32;
            v9 = v28;
          }
          while ( v18 != v8 );
          v7 = *((unsigned int *)a2 + 2);
          v10 = *(_QWORD **)a1;
        }
      }
      else
      {
        v12 = 32 * v9;
        v13 = (_QWORD *)((char *)v8 + v12);
        while ( v13 != v8 )
        {
          while ( 1 )
          {
            v13 -= 4;
            v12 = (__int64)(v13 + 2);
            if ( (_QWORD *)*v13 == v13 + 2 )
              break;
            v27 = v7;
            j_j___libc_free_0(*v13, v13[2] + 1LL);
            v7 = v27;
            if ( v13 == v8 )
              goto LABEL_8;
          }
        }
LABEL_8:
        *(_DWORD *)(a1 + 8) = 0;
        sub_12BE710(a1, v7, v12, v11, v7, a6);
        v7 = *((unsigned int *)a2 + 2);
        v10 = *(_QWORD **)a1;
        v9 = 0;
      }
      v14 = (_QWORD *)((char *)v10 + v9);
      v15 = *a2 + 32 * v7;
      for ( i = v9 + *a2; v15 != i; v14 += 4 )
      {
        if ( v14 )
        {
          *v14 = (__int64)(v14 + 2);
          sub_20A0C10(v14, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
        }
        i += 32LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v26;
  }
}
