// Function: sub_240DBB0
// Address: 0x240dbb0
//
void __fastcall sub_240DBB0(__int64 a1, char *a2, _BYTE *a3)
{
  size_t v4; // r12
  _BYTE *v5; // r8
  char *v6; // r13
  __int64 v7; // rax
  char *v8; // rcx
  size_t v9; // rdx
  _BYTE *v10; // rax
  char *v11; // r12
  char *v12; // r12
  _BYTE *src; // [rsp+8h] [rbp-38h]

  v4 = a3 - a2;
  v5 = *(_BYTE **)a1;
  if ( (unsigned __int64)(a3 - a2) <= *(_QWORD *)(a1 + 16) - *(_QWORD *)a1 )
  {
    v8 = *(char **)(a1 + 8);
    v9 = v8 - v5;
    if ( v4 > v8 - v5 )
    {
      v12 = &a2[v9];
      if ( a2 != &a2[v9] )
      {
        memmove(*(void **)a1, a2, v9);
        v8 = *(char **)(a1 + 8);
      }
      if ( a3 != v12 )
        v8 = (char *)memmove(v8, v12, a3 - v12);
      *(_QWORD *)(a1 + 8) = &v8[a3 - v12];
    }
    else
    {
      if ( a3 != a2 )
      {
        v10 = memmove(*(void **)a1, a2, v4);
        v8 = *(char **)(a1 + 8);
        v5 = v10;
      }
      v11 = &v5[v4];
      if ( v11 != v8 )
        *(_QWORD *)(a1 + 8) = v11;
    }
  }
  else
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v6 = 0;
    if ( v4 )
    {
      v7 = sub_22077B0(a3 - a2);
      v5 = *(_BYTE **)a1;
      v6 = (char *)v7;
    }
    if ( a3 != a2 )
    {
      src = v5;
      memcpy(v6, a2, v4);
      v5 = src;
    }
    if ( v5 )
      j_j___libc_free_0((unsigned __int64)v5);
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = &v6[v4];
    *(_QWORD *)(a1 + 16) = &v6[v4];
  }
}
