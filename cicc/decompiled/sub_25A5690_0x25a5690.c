// Function: sub_25A5690
// Address: 0x25a5690
//
void __fastcall sub_25A5690(__int64 a1, char **a2)
{
  char *v3; // rdx
  char *v4; // r15
  char *v5; // r14
  size_t v6; // r13
  _BYTE *v7; // rdi
  char *v8; // r8
  char *v9; // r13
  __int64 v10; // rax
  char *v11; // r12
  char *v12; // [rsp-40h] [rbp-40h]

  if ( a2 != (char **)a1 )
  {
    v3 = a2[1];
    v4 = *a2;
    v5 = *(char **)a1;
    v6 = v3 - *a2;
    if ( v6 > *(_QWORD *)(a1 + 16) - *(_QWORD *)a1 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, a2, v3);
        v12 = a2[1];
        v10 = sub_22077B0(v12 - *a2);
        v5 = *(char **)a1;
        v3 = v12;
        v11 = (char *)v10;
      }
      else
      {
        v11 = 0;
      }
      if ( v3 != v4 )
        memcpy(v11, v4, v6);
      if ( v5 )
        j_j___libc_free_0((unsigned __int64)v5);
      v9 = &v11[v6];
      *(_QWORD *)a1 = v11;
      *(_QWORD *)(a1 + 16) = v9;
      goto LABEL_7;
    }
    v7 = *(_BYTE **)(a1 + 8);
    v8 = (char *)(v7 - v5);
    if ( v6 > v7 - v5 )
    {
      if ( v8 )
      {
        memmove(v5, *a2, v7 - v5);
        v7 = *(_BYTE **)(a1 + 8);
        v5 = *(char **)a1;
        v3 = a2[1];
        v4 = *a2;
        v8 = &v7[-*(_QWORD *)a1];
      }
      if ( &v8[(_QWORD)v4] != v3 )
      {
        memmove(v7, &v8[(_QWORD)v4], v3 - &v8[(_QWORD)v4]);
        v9 = (char *)(*(_QWORD *)a1 + v6);
        goto LABEL_7;
      }
    }
    else if ( v3 != v4 )
    {
      memmove(v5, *a2, a2[1] - *a2);
      v5 = *(char **)a1;
    }
    v9 = &v5[v6];
LABEL_7:
    *(_QWORD *)(a1 + 8) = v9;
  }
}
