// Function: sub_E11180
// Address: 0xe11180
//
__int64 __fastcall sub_E11180(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  _DWORD *v7; // rdx
  int v8; // r15d
  __int64 v9; // rcx
  _DWORD *v10; // r14
  _QWORD *v11; // r12
  unsigned __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r12
  char v15; // al
  __int64 **v16; // rbx
  __int64 *v17; // r13
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rcx
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 n; // [rsp+8h] [rbp-38h]

  v6 = (int)a2 + 196LL;
  v7 = *a1;
  v8 = *((_DWORD *)*a1 + v6 + 2);
  v9 = (unsigned int)(v8 + 1);
  *((_DWORD *)*a1 + v6 + 2) = v9;
  v10 = *a1;
  v11 = (_QWORD *)(*a1)[614];
  v12 = v11[1] + 32LL;
  if ( v12 > 0xFEF )
  {
    v13 = (_QWORD *)malloc(4096, a2, v7, v9, a5, a6);
    if ( !v13 )
      sub_2207530(4096, a2, 0);
    *v13 = v11;
    v11 = v13;
    v13[1] = 0;
    *((_QWORD *)v10 + 614) = v13;
    v12 = v13[1] + 32LL;
  }
  v11[1] = v12;
  v14 = *((_QWORD *)v10 + 614) + *(_QWORD *)(*((_QWORD *)v10 + 614) + 8LL) - 16LL;
  *(_WORD *)(v14 + 8) = 16417;
  v15 = *(_BYTE *)(v14 + 10);
  *(_DWORD *)(v14 + 12) = a2;
  *(_DWORD *)(v14 + 16) = v8;
  *(_BYTE *)(v14 + 10) = v15 & 0xF0 | 5;
  *(_QWORD *)v14 = &unk_49DFA28;
  v16 = (__int64 **)*a1[1];
  if ( v16 )
  {
    v17 = v16[1];
    if ( v17 != v16[2] )
    {
LABEL_6:
      v16[1] = v17 + 1;
      *v17 = v14;
      return v14;
    }
    v19 = *v16;
    n = (char *)v17 - (char *)*v16;
    if ( *v16 == (__int64 *)(v16 + 3) )
    {
      v23 = (__int64 *)malloc(16 * (n >> 3), a2, n, 16 * (n >> 3), a5, a6);
      v22 = v23;
      if ( v23 )
      {
        v21 = n;
        if ( v17 != v19 )
        {
          v24 = (__int64 *)memmove(v23, v19, n);
          v21 = n;
          v22 = v24;
        }
        *v16 = v22;
        goto LABEL_10;
      }
    }
    else
    {
      v20 = realloc(v19);
      v21 = n;
      *v16 = (__int64 *)v20;
      v22 = (__int64 *)v20;
      if ( v20 )
      {
LABEL_10:
        v17 = (__int64 *)((char *)v22 + v21);
        v16[2] = &v22[2 * (n >> 3)];
        goto LABEL_6;
      }
    }
    abort();
  }
  return v14;
}
