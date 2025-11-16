// Function: sub_37F4FD0
// Address: 0x37f4fd0
//
__int64 __fastcall sub_37F4FD0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned __int64 *v4; // r12
  char *v5; // rdx
  _BYTE *v6; // rdi
  char *v7; // r15
  size_t v8; // r13
  _BYTE *v9; // r9
  char *v10; // r8
  char *v11; // r13
  _DWORD *v12; // rcx
  _DWORD *i; // rax
  __int64 result; // rax
  __int64 v15; // rax
  char *v16; // rcx
  char *v17; // [rsp+8h] [rbp-38h]
  char *v18; // [rsp+8h] [rbp-38h]

  v3 = 24LL * *(unsigned int *)(a2 + 24);
  v4 = (unsigned __int64 *)(v3 + *(_QWORD *)(a1 + 344));
  if ( v4 != (unsigned __int64 *)(a1 + 320) )
  {
    v5 = *(char **)(a1 + 328);
    v6 = (_BYTE *)*v4;
    v7 = *(char **)(a1 + 320);
    v8 = *(_QWORD *)(a1 + 328) - (_QWORD)v7;
    if ( v8 > v4[2] - *v4 )
    {
      if ( v8 )
      {
        if ( v8 > 0x7FFFFFFFFFFFFFFCLL )
          sub_4261EA(v6, a2, v5);
        v17 = *(char **)(a1 + 328);
        v15 = sub_22077B0(v17 - v7);
        v5 = v17;
        v16 = (char *)v15;
      }
      else
      {
        v16 = 0;
      }
      if ( v5 != v7 )
        v16 = (char *)memcpy(v16, v7, v8);
      if ( *v4 )
      {
        v18 = v16;
        j_j___libc_free_0(*v4);
        v16 = v18;
      }
      v11 = &v16[v8];
      *v4 = (unsigned __int64)v16;
      v4[2] = (unsigned __int64)v11;
      goto LABEL_7;
    }
    v9 = (_BYTE *)v4[1];
    v10 = (char *)(v9 - v6);
    if ( v8 > v9 - v6 )
    {
      if ( v10 )
      {
        memmove(v6, *(const void **)(a1 + 320), v4[1] - (_QWORD)v6);
        v9 = (_BYTE *)v4[1];
        v6 = (_BYTE *)*v4;
        v5 = *(char **)(a1 + 328);
        v7 = *(char **)(a1 + 320);
        v10 = &v9[-*v4];
      }
      if ( &v10[(_QWORD)v7] != v5 )
      {
        memmove(v9, &v10[(_QWORD)v7], v5 - &v10[(_QWORD)v7]);
        v11 = (char *)(*v4 + v8);
        goto LABEL_7;
      }
    }
    else if ( v5 != v7 )
    {
      memmove(v6, *(const void **)(a1 + 320), *(_QWORD *)(a1 + 328) - (_QWORD)v7);
      v6 = (_BYTE *)*v4;
    }
    v11 = &v6[v8];
LABEL_7:
    v4[1] = (unsigned __int64)v11;
    v4 = (unsigned __int64 *)(*(_QWORD *)(a1 + 344) + v3);
  }
  v12 = (_DWORD *)v4[1];
  for ( i = (_DWORD *)*v4; v12 != i; ++i )
  {
    if ( *i != *(_DWORD *)(a1 + 640) )
      *i -= *(_DWORD *)(a1 + 456);
  }
  result = *(_QWORD *)(a1 + 320);
  if ( result != *(_QWORD *)(a1 + 328) )
    *(_QWORD *)(a1 + 328) = result;
  return result;
}
