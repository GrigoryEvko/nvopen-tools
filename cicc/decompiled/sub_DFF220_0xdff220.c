// Function: sub_DFF220
// Address: 0xdff220
//
__int64 __fastcall sub_DFF220(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  unsigned __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  __int64 v11; // rax
  __int64 v12; // rbx
  char *v13; // rax
  char *v14; // r13
  char *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r12d
  __int64 v21[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  unsigned int v24; // [rsp+20h] [rbp-40h]
  void *src; // [rsp+28h] [rbp-38h]
  char *v26; // [rsp+30h] [rbp-30h]
  char *v27; // [rsp+38h] [rbp-28h]

  v2 = *a1;
  v21[1] = 0;
  v3 = *(_QWORD *)(v2 + 40) - *(_QWORD *)(v2 + 32);
  v21[0] = v2;
  v4 = (unsigned int)(v3 >> 3) | ((unsigned __int64)(unsigned int)(v3 >> 3) >> 1);
  v5 = (((v4 >> 2) | v4) >> 4) | (v4 >> 2) | v4;
  v6 = (((v5 >> 8) | v5) >> 16) | (v5 >> 8) | v5;
  if ( (_DWORD)v6 == -1 )
  {
    v22 = 0;
    v23 = 0;
    v24 = 0;
  }
  else
  {
    v7 = 4 * ((int)v6 + 1) / 3u;
    v8 = ((((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
          | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
          | (v7 + 1)
          | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
        | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
        | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
        | (v7 + 1)
        | ((unsigned __int64)(v7 + 1) >> 1)) >> 16;
    v24 = (v8
         | (((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
           | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
           | (v7 + 1)
           | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
         | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2) | (v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
         | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
         | (v7 + 1)
         | ((v7 + 1) >> 1))
        + 1;
    v9 = (_QWORD *)sub_C7D670(
                     16
                   * ((v8
                     | (((((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
                         | (v7 + 1)
                         | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
                       | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
                       | (v7 + 1)
                       | ((unsigned __int64)(v7 + 1) >> 1)) >> 8)
                     | (((((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
                       | (v7 + 1)
                       | ((unsigned __int64)(v7 + 1) >> 1)) >> 4)
                     | (((v7 + 1) | ((unsigned __int64)(v7 + 1) >> 1)) >> 2)
                     | (v7 + 1)
                     | ((unsigned __int64)(v7 + 1) >> 1))
                    + 1),
                     8);
    v23 = 0;
    v22 = v9;
    for ( i = &v9[2 * v24]; i != v9; v9 += 2 )
    {
      if ( v9 )
        *v9 = -4096;
    }
  }
  src = 0;
  v26 = 0;
  v27 = 0;
  v11 = (__int64)(*(_QWORD *)(v2 + 40) - *(_QWORD *)(v2 + 32)) >> 3;
  if ( (_DWORD)v11 )
  {
    v12 = 8LL * (unsigned int)v11;
    v13 = (char *)sub_22077B0(v12);
    v14 = v13;
    if ( v26 - (_BYTE *)src > 0 )
    {
      memmove(v13, src, v26 - (_BYTE *)src);
      j_j___libc_free_0(src, v27 - (_BYTE *)src);
    }
    src = v14;
    v26 = v14;
    v27 = &v14[v12];
  }
  sub_D4E470(v21, a2);
  v19 = sub_DFEF30((__int64)v21, a2, v15, v16, v17, v18) ^ 1;
  if ( src )
    j_j___libc_free_0(src, v27 - (_BYTE *)src);
  sub_C7D6A0((__int64)v22, 16LL * v24, 8);
  return v19;
}
