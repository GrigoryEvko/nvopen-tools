// Function: sub_21D74F0
// Address: 0x21d74f0
//
__int64 __fastcall sub_21D74F0(__int64 a1, _QWORD *a2, int a3, unsigned __int8 a4, __int64 a5)
{
  __int64 *v8; // rax
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _BYTE *v11; // r8
  void **v12; // rdi
  __int64 v13; // r13
  char *v14; // r15
  __int64 *v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r12
  size_t *v23; // rsi
  size_t v24; // r15
  char *v25; // rsi
  char *s[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v29[2]; // [rsp+30h] [rbp-70h] BYREF
  void *v30; // [rsp+40h] [rbp-60h] BYREF
  void *v31; // [rsp+48h] [rbp-58h]
  __int64 v32; // [rsp+50h] [rbp-50h]
  void *dest; // [rsp+58h] [rbp-48h]
  int v34; // [rsp+60h] [rbp-40h]
  char **v35; // [rsp+68h] [rbp-38h]

  v35 = s;
  v8 = (__int64 *)a2[4];
  s[0] = (char *)v29;
  v9 = *(_QWORD *)(a1 + 8);
  s[1] = 0;
  LOBYTE(v29[0]) = 0;
  v34 = 1;
  dest = 0;
  v32 = 0;
  v31 = 0;
  v30 = &unk_49EFBE0;
  v10 = (_BYTE *)sub_17007B0(v9, *v8);
  if ( (*v10 & 4) != 0 )
  {
    v23 = (size_t *)*((_QWORD *)v10 - 1);
    v11 = dest;
    v24 = *v23;
    v25 = (char *)(v23 + 2);
    if ( v32 - (__int64)dest < v24 )
    {
      v12 = (void **)sub_16E7EE0((__int64)&v30, v25, v24);
      v11 = v12[3];
      if ( (unsigned __int64)((_BYTE *)v12[2] - v11) > 6 )
        goto LABEL_4;
      goto LABEL_18;
    }
    if ( v24 )
    {
      memcpy(dest, v25, v24);
      v12 = &v30;
      dest = (char *)dest + v24;
      v11 = dest;
    }
    else
    {
      v12 = &v30;
    }
  }
  else
  {
    v11 = dest;
    v12 = &v30;
  }
  if ( (unsigned __int64)(v32 - (_QWORD)v11) > 6 )
  {
LABEL_4:
    *(_DWORD *)v11 = 1918988383;
    *((_WORD *)v11 + 2) = 28001;
    v11[6] = 95;
    v12[3] = (char *)v12[3] + 7;
    goto LABEL_5;
  }
LABEL_18:
  v12 = (void **)sub_16E7EE0((__int64)v12, "_param_", 7u);
LABEL_5:
  sub_16E7AB0((__int64)v12, a3);
  if ( dest != v31 )
    sub_16E7BA0((__int64 *)&v30);
  v13 = *(_QWORD *)(a1 + 81544);
  v14 = s[0];
  v15 = (__int64 *)sub_22077B0(32);
  v18 = v15;
  if ( v15 )
  {
    v19 = -1;
    *v15 = (__int64)(v15 + 2);
    if ( v14 )
      v19 = (__int64)&v14[strlen(v14)];
    sub_21CA7A0(v18, v14, v19);
  }
  v20 = *(unsigned int *)(v13 + 83296);
  if ( (unsigned int)v20 >= *(_DWORD *)(v13 + 83300) )
  {
    sub_16CD150(v13 + 83288, (const void *)(v13 + 83304), 0, 8, v16, v17);
    v20 = *(unsigned int *)(v13 + 83296);
  }
  *(_QWORD *)(*(_QWORD *)(v13 + 83288) + 8 * v20) = v18;
  ++*(_DWORD *)(v13 + 83296);
  v21 = sub_1D2F9D0(a2, (const char *)*v18, a4, a5, 0);
  sub_16E7BC0((__int64 *)&v30);
  if ( (_QWORD *)s[0] != v29 )
    j_j___libc_free_0(s[0], v29[0] + 1LL);
  return v21;
}
