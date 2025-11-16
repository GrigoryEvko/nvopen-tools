// Function: sub_3060640
// Address: 0x3060640
//
unsigned __int64 __fastcall sub_3060640(unsigned __int64 *a1, unsigned __int64 *a2, _QWORD *a3)
{
  unsigned __int64 v3; // r14
  __int64 v4; // rax
  bool v6; // zf
  __int64 v8; // rsi
  __int64 v9; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rbx
  char *v15; // rdx
  _QWORD *v16; // rbx
  unsigned __int64 *v17; // r15
  __int64 v18; // rcx
  unsigned __int64 v19; // rdi
  void (*v20)(void); // rcx
  void *v21; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-50h]
  unsigned __int64 v27; // [rsp+18h] [rbp-48h]
  unsigned __int64 v28; // [rsp+20h] [rbp-40h]
  unsigned __int64 v29; // [rsp+28h] [rbp-38h]

  v3 = *a1;
  v26 = (unsigned __int64 *)a1[1];
  v4 = (__int64)((__int64)v26 - *a1) >> 3;
  if ( v4 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = v4 == 0;
  v8 = (__int64)((__int64)v26 - v3) >> 3;
  v9 = 1;
  if ( !v6 )
    v9 = (__int64)((__int64)v26 - v3) >> 3;
  v11 = __CFADD__(v8, v9);
  v12 = v8 + v9;
  v13 = (char *)a2 - v3;
  if ( v11 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v27 = 0;
      v14 = 8;
      v28 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0xFFFFFFFFFFFFFFFLL )
      v12 = 0xFFFFFFFFFFFFFFFLL;
    v23 = 8 * v12;
  }
  v25 = a3;
  v24 = sub_22077B0(v23);
  v13 = (char *)a2 - v3;
  a3 = v25;
  v28 = v24;
  v27 = v24 + v23;
  v14 = v24 + 8;
LABEL_7:
  v15 = &v13[v28];
  if ( v15 )
    *(_QWORD *)v15 = *a3;
  if ( a2 != (unsigned __int64 *)v3 )
  {
    v16 = (_QWORD *)v28;
    v17 = (unsigned __int64 *)v3;
    while ( 1 )
    {
      v19 = *v17;
      if ( v16 )
        break;
      if ( !v19 )
        goto LABEL_12;
      v20 = *(void (**)(void))(*(_QWORD *)v19 + 8LL);
      if ( (char *)v20 != (char *)sub_305C760 )
      {
        v20();
LABEL_12:
        ++v17;
        v18 = (__int64)(v16 + 1);
        if ( a2 == v17 )
          goto LABEL_18;
        goto LABEL_13;
      }
      v29 = *v17++;
      *(_QWORD *)v19 = &unk_4A30E28;
      nullsub_181();
      j_j___libc_free_0(v29);
      v18 = 8;
      if ( a2 == v17 )
      {
LABEL_18:
        v14 = (__int64)(v16 + 2);
        goto LABEL_19;
      }
LABEL_13:
      v16 = (_QWORD *)v18;
    }
    *v16 = v19;
    *v17 = 0;
    goto LABEL_12;
  }
LABEL_19:
  if ( a2 != v26 )
  {
    v21 = (void *)v14;
    v14 += (char *)v26 - (char *)a2;
    memcpy(v21, a2, (char *)v26 - (char *)a2);
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  a1[1] = v14;
  *a1 = v28;
  a1[2] = v27;
  return v27;
}
