// Function: sub_18BD7E0
// Address: 0x18bd7e0
//
_QWORD *__fastcall sub_18BD7E0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  _QWORD *v9; // r12
  char **v10; // r9
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  unsigned __int64 v13; // r8
  char *v14; // r13
  const void *v15; // rax
  const void *v16; // rsi
  signed __int64 v17; // r15
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  char *v20; // rcx
  unsigned __int64 v21; // r8
  _QWORD *v22; // r14
  _BOOL8 v23; // rdi
  _QWORD *v25; // rax
  signed __int64 v26; // rsi
  _QWORD *v27; // rax
  unsigned __int64 v28; // [rsp+8h] [rbp-48h]
  char **v29; // [rsp+10h] [rbp-40h]
  char **v30; // [rsp+10h] [rbp-40h]
  unsigned __int64 v31; // [rsp+10h] [rbp-40h]
  unsigned __int64 v32; // [rsp+18h] [rbp-38h]

  v6 = (_QWORD *)sub_22077B0(80);
  v8 = *a3;
  v9 = v6;
  v10 = (char **)(v6 + 4);
  v11 = *(_QWORD *)(*a3 + 8) - *(_QWORD *)*a3;
  v6[4] = 0;
  v6[5] = 0;
  v6[6] = 0;
  if ( v11 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(80, a2, v7);
    v29 = (char **)(v6 + 4);
    v32 = v11;
    v12 = sub_22077B0(v11);
    v13 = v32;
    v10 = v29;
    v14 = (char *)v12;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  v9[4] = v14;
  v9[5] = v14;
  v9[6] = &v14[v13];
  v15 = *(const void **)(v8 + 8);
  v16 = *(const void **)v8;
  v17 = (signed __int64)v15 - *(_QWORD *)v8;
  if ( v15 != v16 )
  {
    v28 = v13;
    v30 = v10;
    memmove(v14, v16, v17);
    *((_DWORD *)v9 + 14) = 0;
    v9[5] = &v14[v17];
    v9[8] = 0;
    v9[9] = 0;
    v18 = sub_14F7820(a1, a2, v30);
    v20 = &v14[v17];
    v21 = v28;
    v22 = v18;
    if ( v19 )
      goto LABEL_6;
    goto LABEL_19;
  }
  v31 = v13;
  v9[5] = &v14[v17];
  *((_DWORD *)v9 + 14) = 0;
  v9[8] = 0;
  v9[9] = 0;
  v27 = sub_14F7820(a1, a2, v10);
  v20 = &v14[v17];
  v21 = v31;
  v22 = v27;
  if ( !v19 )
  {
    if ( !v14 )
    {
LABEL_20:
      j_j___libc_free_0(v9, 80);
      return v22;
    }
LABEL_19:
    j_j___libc_free_0(v14, v21);
    goto LABEL_20;
  }
LABEL_6:
  v23 = 1;
  if ( !v22 && v19 != a1 + 1 )
  {
    v25 = (_QWORD *)v19[4];
    v26 = v19[5] - (_QWORD)v25;
    if ( v17 > v26 )
      v20 = &v14[v26];
    if ( v14 == v20 )
    {
LABEL_22:
      v23 = v19[5] != (_QWORD)v25;
    }
    else
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v14 < *v25 )
        {
          v23 = 1;
          goto LABEL_7;
        }
        if ( *(_QWORD *)v14 > *v25 )
          break;
        v14 += 8;
        ++v25;
        if ( v20 == v14 )
          goto LABEL_22;
      }
      v23 = 0;
    }
  }
LABEL_7:
  sub_220F040(v23, v9, v19, a1 + 1);
  ++a1[5];
  return v9;
}
