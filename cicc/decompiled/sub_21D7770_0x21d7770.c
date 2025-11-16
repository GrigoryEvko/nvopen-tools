// Function: sub_21D7770
// Address: 0x21d7770
//
__int64 *__fastcall sub_21D7770(__int64 a1, __int64 *a2, int a3)
{
  __int64 v5; // rdi
  __int64 v6; // rsi
  _BYTE *v7; // rax
  char *v8; // r8
  void **v9; // rdi
  __int64 v10; // rbx
  char *v11; // r15
  __int64 *v12; // rax
  int v13; // r8d
  int v14; // r9d
  __int64 *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  size_t *v19; // rsi
  size_t v20; // r15
  char *v21; // rsi
  char *s[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v23[2]; // [rsp+10h] [rbp-70h] BYREF
  void *v24; // [rsp+20h] [rbp-60h] BYREF
  void *v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  void *dest; // [rsp+38h] [rbp-48h]
  int v28; // [rsp+40h] [rbp-40h]
  char **v29; // [rsp+48h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *a2;
  s[0] = (char *)v23;
  s[1] = 0;
  LOBYTE(v23[0]) = 0;
  v28 = 1;
  dest = 0;
  v26 = 0;
  v25 = 0;
  v24 = &unk_49EFBE0;
  v29 = s;
  v7 = (_BYTE *)sub_17007B0(v5, v6);
  if ( (*v7 & 4) != 0 )
  {
    v19 = (size_t *)*((_QWORD *)v7 - 1);
    v8 = (char *)dest;
    v20 = *v19;
    v21 = (char *)(v19 + 2);
    if ( v26 - (__int64)dest < v20 )
    {
      v9 = (void **)sub_16E7EE0((__int64)&v24, v21, v20);
      v8 = (char *)v9[3];
      if ( (unsigned __int64)((_BYTE *)v9[2] - v8) > 6 )
        goto LABEL_4;
      goto LABEL_18;
    }
    if ( v20 )
    {
      memcpy(dest, v21, v20);
      v9 = &v24;
      v8 = (char *)dest + v20;
      dest = (char *)dest + v20;
    }
    else
    {
      v9 = &v24;
    }
  }
  else
  {
    v8 = (char *)dest;
    v9 = &v24;
  }
  if ( (unsigned __int64)(v26 - (_QWORD)v8) > 6 )
  {
LABEL_4:
    *(_DWORD *)v8 = 1918988383;
    *((_WORD *)v8 + 2) = 28001;
    v8[6] = 95;
    v9[3] = (char *)v9[3] + 7;
    goto LABEL_5;
  }
LABEL_18:
  v9 = (void **)sub_16E7EE0((__int64)v9, "_param_", 7u);
LABEL_5:
  sub_16E7AB0((__int64)v9, a3);
  if ( dest != v25 )
    sub_16E7BA0((__int64 *)&v24);
  v10 = *(_QWORD *)(a1 + 81544);
  v11 = s[0];
  v12 = (__int64 *)sub_22077B0(32);
  v15 = v12;
  if ( v12 )
  {
    v16 = -1;
    *v12 = (__int64)(v12 + 2);
    if ( v11 )
      v16 = (__int64)&v11[strlen(v11)];
    sub_21CA7A0(v15, v11, v16);
  }
  v17 = *(unsigned int *)(v10 + 83296);
  if ( (unsigned int)v17 >= *(_DWORD *)(v10 + 83300) )
  {
    sub_16CD150(v10 + 83288, (const void *)(v10 + 83304), 0, 8, v13, v14);
    v17 = *(unsigned int *)(v10 + 83296);
  }
  *(_QWORD *)(*(_QWORD *)(v10 + 83288) + 8 * v17) = v15;
  ++*(_DWORD *)(v10 + 83296);
  sub_16E7BC0((__int64 *)&v24);
  if ( (_QWORD *)s[0] != v23 )
    j_j___libc_free_0(s[0], v23[0] + 1LL);
  return v15;
}
