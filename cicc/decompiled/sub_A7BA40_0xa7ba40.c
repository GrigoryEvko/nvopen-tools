// Function: sub_A7BA40
// Address: 0xa7ba40
//
void __fastcall sub_A7BA40(__int64 a1, char *a2, _BYTE *a3)
{
  size_t v4; // r12
  _BYTE *v5; // r8
  unsigned __int64 v6; // r15
  char *v7; // r13
  __int64 v8; // rax
  char *v9; // rcx
  size_t v10; // rdx
  _BYTE *v11; // rax
  char *v12; // r12
  char *v13; // r12
  _BYTE *src; // [rsp+8h] [rbp-38h]

  v4 = a3 - a2;
  v5 = *(_BYTE **)a1;
  v6 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
  if ( a3 - a2 <= v6 )
  {
    v9 = *(char **)(a1 + 8);
    v10 = v9 - v5;
    if ( v4 > v9 - v5 )
    {
      v13 = &a2[v10];
      if ( a2 != &a2[v10] )
      {
        memmove(*(void **)a1, a2, v10);
        v9 = *(char **)(a1 + 8);
      }
      if ( a3 != v13 )
        v9 = (char *)memmove(v9, v13, a3 - v13);
      *(_QWORD *)(a1 + 8) = &v9[a3 - v13];
    }
    else
    {
      if ( a3 != a2 )
      {
        v11 = memmove(*(void **)a1, a2, v4);
        v9 = *(char **)(a1 + 8);
        v5 = v11;
      }
      v12 = &v5[v4];
      if ( v12 != v9 )
        *(_QWORD *)(a1 + 8) = v12;
    }
  }
  else
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v7 = 0;
    if ( v4 )
    {
      v8 = sub_22077B0(a3 - a2);
      v5 = *(_BYTE **)a1;
      v7 = (char *)v8;
      v6 = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
    }
    if ( a3 != a2 )
    {
      src = v5;
      memcpy(v7, a2, v4);
      v5 = src;
    }
    if ( v5 )
      j_j___libc_free_0(v5, v6);
    *(_QWORD *)a1 = v7;
    *(_QWORD *)(a1 + 8) = &v7[v4];
    *(_QWORD *)(a1 + 16) = &v7[v4];
  }
}
