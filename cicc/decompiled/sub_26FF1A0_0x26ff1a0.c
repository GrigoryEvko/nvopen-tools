// Function: sub_26FF1A0
// Address: 0x26ff1a0
//
unsigned __int64 __fastcall sub_26FF1A0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  unsigned __int64 v9; // r12
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
  _QWORD *v21; // r14
  char v22; // di
  _QWORD *v24; // rax
  signed __int64 v25; // rsi
  _QWORD *v26; // rax
  char **v27; // [rsp+10h] [rbp-40h]
  char **v28; // [rsp+10h] [rbp-40h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]

  v6 = (_QWORD *)sub_22077B0(0x50u);
  v8 = *a3;
  v9 = (unsigned __int64)v6;
  v10 = (char **)(v6 + 4);
  v11 = *(_QWORD *)(*a3 + 8) - *(_QWORD *)*a3;
  v6[4] = 0;
  v6[5] = 0;
  v6[6] = 0;
  if ( v11 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(80, a2, v7);
    v27 = (char **)(v6 + 4);
    v29 = v11;
    v12 = sub_22077B0(v11);
    v13 = v29;
    v10 = v27;
    v14 = (char *)v12;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  *(_QWORD *)(v9 + 32) = v14;
  *(_QWORD *)(v9 + 40) = v14;
  *(_QWORD *)(v9 + 48) = &v14[v13];
  v15 = *(const void **)(v8 + 8);
  v16 = *(const void **)v8;
  v17 = (signed __int64)v15 - *(_QWORD *)v8;
  if ( v15 != v16 )
  {
    v28 = v10;
    memmove(v14, v16, v17);
    *(_DWORD *)(v9 + 56) = 0;
    *(_QWORD *)(v9 + 40) = &v14[v17];
    *(_QWORD *)(v9 + 64) = 0;
    *(_QWORD *)(v9 + 72) = 0;
    v18 = sub_9D7C50(a1, a2, v28);
    v20 = &v14[v17];
    v21 = v18;
    if ( v19 )
      goto LABEL_6;
    goto LABEL_19;
  }
  *(_QWORD *)(v9 + 40) = &v14[v17];
  *(_DWORD *)(v9 + 56) = 0;
  *(_QWORD *)(v9 + 64) = 0;
  *(_QWORD *)(v9 + 72) = 0;
  v26 = sub_9D7C50(a1, a2, v10);
  v20 = &v14[v17];
  v21 = v26;
  if ( !v19 )
  {
    if ( !v14 )
    {
LABEL_20:
      j_j___libc_free_0(v9);
      return (unsigned __int64)v21;
    }
LABEL_19:
    j_j___libc_free_0((unsigned __int64)v14);
    goto LABEL_20;
  }
LABEL_6:
  v22 = 1;
  if ( !v21 && v19 != a1 + 1 )
  {
    v24 = (_QWORD *)v19[4];
    v25 = v19[5] - (_QWORD)v24;
    if ( v17 > v25 )
      v20 = &v14[v25];
    if ( v14 == v20 )
    {
LABEL_22:
      v22 = v19[5] != (_QWORD)v24;
    }
    else
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v14 < *v24 )
        {
          v22 = 1;
          goto LABEL_7;
        }
        if ( *(_QWORD *)v14 > *v24 )
          break;
        v14 += 8;
        ++v24;
        if ( v20 == v14 )
          goto LABEL_22;
      }
      v22 = 0;
    }
  }
LABEL_7:
  sub_220F040(v22, v9, v19, a1 + 1);
  ++a1[5];
  return v9;
}
