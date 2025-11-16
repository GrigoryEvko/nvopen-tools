// Function: sub_142B670
// Address: 0x142b670
//
void __fastcall sub_142B670(__int64 a1, char **a2)
{
  char *v3; // rdx
  char *v4; // r15
  char *v5; // r14
  size_t v6; // r13
  size_t v7; // r8
  _BYTE *v8; // rdi
  char *v9; // r8
  char *v10; // r13
  __int64 v11; // rax
  char *v12; // r12
  char *v13; // [rsp-40h] [rbp-40h]
  size_t v14; // [rsp-40h] [rbp-40h]

  if ( a2 != (char **)a1 )
  {
    v3 = a2[1];
    v4 = *a2;
    v5 = *(char **)a1;
    v6 = v3 - *a2;
    v7 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    if ( v6 > v7 )
    {
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, a2, v3);
        v13 = a2[1];
        v11 = sub_22077B0(v13 - *a2);
        v5 = *(char **)a1;
        v3 = v13;
        v12 = (char *)v11;
        v7 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
      }
      else
      {
        v12 = 0;
      }
      if ( v3 != v4 )
      {
        v14 = v7;
        memcpy(v12, v4, v6);
        v7 = v14;
      }
      if ( v5 )
        j_j___libc_free_0(v5, v7);
      v10 = &v12[v6];
      *(_QWORD *)a1 = v12;
      *(_QWORD *)(a1 + 16) = v10;
      goto LABEL_7;
    }
    v8 = *(_BYTE **)(a1 + 8);
    v9 = (char *)(v8 - v5);
    if ( v6 > v8 - v5 )
    {
      if ( v9 )
      {
        memmove(v5, *a2, v8 - v5);
        v8 = *(_BYTE **)(a1 + 8);
        v5 = *(char **)a1;
        v3 = a2[1];
        v4 = *a2;
        v9 = &v8[-*(_QWORD *)a1];
      }
      if ( &v9[(_QWORD)v4] != v3 )
      {
        memmove(v8, &v9[(_QWORD)v4], v3 - &v9[(_QWORD)v4]);
        v10 = (char *)(*(_QWORD *)a1 + v6);
        goto LABEL_7;
      }
    }
    else if ( v3 != v4 )
    {
      memmove(v5, *a2, a2[1] - *a2);
      v5 = *(char **)a1;
    }
    v10 = &v5[v6];
LABEL_7:
    *(_QWORD *)(a1 + 8) = v10;
  }
}
