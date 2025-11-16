// Function: sub_15625F0
// Address: 0x15625f0
//
_QWORD *__fastcall sub_15625F0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  void *v7; // rbx
  _QWORD *v8; // r13
  size_t v9; // r12
  __int64 v10; // r15
  size_t v11; // r14
  size_t v12; // rdx
  int v13; // eax
  __int64 v14; // r14
  size_t v15; // r15
  size_t v16; // rcx
  size_t v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  _BYTE *v20; // rsi
  size_t v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rdx
  _QWORD *v24; // r8
  __int64 v25; // rdi
  __int64 v27; // rdi
  size_t v28; // rcx
  size_t v29; // r9
  size_t v30; // rdx
  int v31; // eax
  unsigned int v32; // edi
  __int64 v33; // rcx
  _QWORD *v34; // [rsp+10h] [rbp-C0h]
  size_t v35; // [rsp+10h] [rbp-C0h]
  __int64 v36; // [rsp+18h] [rbp-B8h]
  size_t v37; // [rsp+18h] [rbp-B8h]
  _QWORD *v38; // [rsp+28h] [rbp-A8h]
  _QWORD *v39; // [rsp+30h] [rbp-A0h]
  size_t v40; // [rsp+30h] [rbp-A0h]
  __int64 v41; // [rsp+30h] [rbp-A0h]
  __int64 v42; // [rsp+30h] [rbp-A0h]
  __int64 v43; // [rsp+30h] [rbp-A0h]
  _QWORD *v44; // [rsp+30h] [rbp-A0h]
  _QWORD *v46; // [rsp+58h] [rbp-78h]
  void *s2; // [rsp+60h] [rbp-70h] BYREF
  size_t v48; // [rsp+68h] [rbp-68h]
  _QWORD v49[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v50[2]; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v51[8]; // [rsp+90h] [rbp-40h] BYREF

  if ( !a1[7] )
    a1[7] = a2[7];
  if ( !a1[8] )
    a1[8] = a2[8];
  if ( !a1[9] )
    a1[9] = a2[9];
  if ( !a1[10] )
    a1[10] = a2[10];
  if ( !a1[11] )
    a1[11] = a2[11];
  *a1 |= *a2;
  v2 = (_QWORD *)a2[4];
  v38 = a2 + 2;
  if ( v2 == a2 + 2 )
    return a1;
  v46 = a1 + 2;
  do
  {
    v3 = (_BYTE *)v2[4];
    v4 = (__int64)&v3[v2[5]];
    s2 = v49;
    sub_155CB60((__int64 *)&s2, v3, v4);
    v5 = (_BYTE *)v2[8];
    v6 = (__int64)&v5[v2[9]];
    v50[0] = (__int64)v51;
    sub_155CB60(v50, v5, v6);
    if ( !a1[3] )
    {
      v8 = a1 + 2;
      goto LABEL_34;
    }
    v7 = s2;
    v39 = v2;
    v8 = a1 + 2;
    v9 = v48;
    v10 = a1[3];
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(v10 + 40);
        v12 = v9;
        if ( v11 <= v9 )
          v12 = *(_QWORD *)(v10 + 40);
        if ( v12 )
        {
          v13 = memcmp(*(const void **)(v10 + 32), v7, v12);
          if ( v13 )
            break;
        }
        v14 = v11 - v9;
        if ( v14 >= 0x80000000LL )
          goto LABEL_24;
        if ( v14 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v13 = v14;
          break;
        }
LABEL_15:
        v10 = *(_QWORD *)(v10 + 24);
        if ( !v10 )
          goto LABEL_25;
      }
      if ( v13 < 0 )
        goto LABEL_15;
LABEL_24:
      v8 = (_QWORD *)v10;
      v10 = *(_QWORD *)(v10 + 16);
    }
    while ( v10 );
LABEL_25:
    v15 = v9;
    v2 = v39;
    if ( v46 == v8 )
      goto LABEL_34;
    v16 = v8[5];
    v17 = v15;
    if ( v16 <= v15 )
      v17 = v8[5];
    if ( v17 && (v40 = v8[5], v18 = memcmp(v7, (const void *)v8[4], v17), v16 = v40, v18) )
    {
LABEL_33:
      if ( v18 < 0 )
        goto LABEL_34;
    }
    else if ( (__int64)(v15 - v16) < 0x80000000LL )
    {
      if ( (__int64)(v15 - v16) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v18 = v15 - v16;
        goto LABEL_33;
      }
LABEL_34:
      v34 = v8;
      v19 = sub_22077B0(96);
      v20 = s2;
      v21 = v48;
      v8 = (_QWORD *)v19;
      *(_QWORD *)(v19 + 32) = v19 + 48;
      v36 = v19 + 48;
      v41 = v19 + 32;
      sub_155CB60((__int64 *)(v19 + 32), v20, (__int64)&v20[v21]);
      v8[8] = v8 + 10;
      v8[9] = 0;
      *((_BYTE *)v8 + 80) = 0;
      v22 = sub_1562360(a1 + 1, v34, v41);
      v24 = v23;
      if ( v23 )
      {
        if ( v46 == v23 || v22 )
        {
LABEL_37:
          v25 = 1;
          goto LABEL_38;
        }
        v28 = v8[5];
        v30 = v23[5];
        v29 = v30;
        if ( v28 <= v30 )
          v30 = v8[5];
        if ( !v30 )
          goto LABEL_53;
        v35 = v29;
        v37 = v8[5];
        v44 = v24;
        v31 = memcmp((const void *)v8[4], (const void *)v24[4], v30);
        v24 = v44;
        v28 = v37;
        v29 = v35;
        v32 = v31;
        if ( v31 )
        {
LABEL_56:
          v25 = v32 >> 31;
        }
        else
        {
LABEL_53:
          v33 = v28 - v29;
          v25 = 0;
          if ( v33 < 0x80000000LL )
          {
            if ( v33 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_37;
            v32 = v33;
            goto LABEL_56;
          }
        }
LABEL_38:
        sub_220F040(v25, v8, v24, v46);
        ++a1[6];
      }
      else
      {
        v27 = v8[4];
        if ( v36 != v27 )
        {
          v42 = v22;
          j_j___libc_free_0(v27, v8[6] + 1LL);
          v22 = v42;
        }
        v43 = v22;
        j_j___libc_free_0(v8, 96);
        v8 = (_QWORD *)v43;
      }
    }
    sub_2240AE0(v8 + 8, v50);
    if ( (_QWORD *)v50[0] != v51 )
      j_j___libc_free_0(v50[0], v51[0] + 1LL);
    if ( s2 != v49 )
      j_j___libc_free_0(s2, v49[0] + 1LL);
    v2 = (_QWORD *)sub_220EF30(v2);
  }
  while ( v38 != v2 );
  return a1;
}
