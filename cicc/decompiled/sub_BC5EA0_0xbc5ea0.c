// Function: sub_BC5EA0
// Address: 0xbc5ea0
//
_QWORD *__fastcall sub_BC5EA0(_QWORD *a1, char *a2, char *a3, unsigned __int64 a4)
{
  _QWORD *v4; // rax
  char *v5; // r15
  _QWORD *v6; // r14
  _QWORD *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // r12
  void *v13; // rax
  void *v14; // rcx
  _QWORD *result; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r12
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rcx
  __int64 v21; // r13
  _QWORD *v22; // rbx
  _QWORD *v23; // r9
  _QWORD *v24; // r13
  unsigned __int64 v25; // r14
  __int64 v26; // rcx
  unsigned __int64 v27; // rsi
  size_t v28; // rdx
  int v29; // eax
  bool v30; // zf
  _QWORD *v31; // rax
  _QWORD *v32; // rbx
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  char v35; // al
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r12
  _QWORD *v38; // r11
  _QWORD *v39; // rdx
  size_t v40; // r13
  void *v41; // rax
  _QWORD *v42; // rax
  _QWORD *v43; // rsi
  unsigned __int64 v44; // rdi
  _QWORD *v45; // rcx
  unsigned __int64 v46; // rdx
  _QWORD **v47; // rax
  __int64 v48; // rdx
  void *v49; // [rsp+8h] [rbp-68h]
  _QWORD *v50; // [rsp+10h] [rbp-60h]
  _QWORD *v51; // [rsp+18h] [rbp-58h]
  _QWORD *v53; // [rsp+28h] [rbp-48h]
  __int64 v54; // [rsp+30h] [rbp-40h]
  _QWORD *v55; // [rsp+30h] [rbp-40h]
  unsigned __int64 v56; // [rsp+38h] [rbp-38h]
  unsigned __int64 v57; // [rsp+38h] [rbp-38h]
  unsigned __int64 v58; // [rsp+38h] [rbp-38h]

  v4 = a1 + 6;
  v5 = a2;
  v6 = a1;
  v7 = a1 + 4;
  *(v7 - 4) = v4;
  v49 = v4;
  *(v7 - 3) = 1;
  *(v7 - 2) = 0;
  *(v7 - 1) = 0;
  *(_DWORD *)v7 = 1065353216;
  v7[1] = 0;
  v7[2] = 0;
  if ( (a3 - a2) >> 5 >= a4 )
    a4 = (a3 - a2) >> 5;
  v51 = v7;
  v8 = a4;
  v9 = sub_222D860(v7, a4);
  if ( v6[1] < v9 )
  {
    v12 = v9;
    if ( v9 == 1 )
    {
      v6[6] = 0;
      v14 = v49;
    }
    else
    {
      if ( v9 > 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_47;
      v13 = (void *)sub_22077B0(8 * v9);
      v14 = memset(v13, 0, 8 * v12);
    }
    *v6 = v14;
    v6[1] = v12;
  }
  result = v6 + 2;
  v50 = v6 + 2;
  while ( a3 != v5 )
  {
    v16 = sub_22076E0(*(_QWORD *)v5, *((_QWORD *)v5 + 1), 3339675911LL);
    v17 = v6[1];
    v18 = v16;
    v19 = v16 % v17;
    v20 = *(_QWORD **)(*v6 + 8 * (v16 % v17));
    v21 = 8 * (v16 % v17);
    if ( !v20 )
      goto LABEL_22;
    v22 = (_QWORD *)*v20;
    v23 = v6;
    v24 = *(_QWORD **)(*v6 + 8 * v19);
    v25 = v16 % v17;
    v26 = 8 * v19;
    v27 = v22[5];
    while ( 1 )
    {
      if ( v18 == v27 )
      {
        v28 = *((_QWORD *)v5 + 1);
        if ( v28 == v22[2] )
        {
          if ( !v28 )
            break;
          v53 = v23;
          v54 = v26;
          v56 = v18;
          v29 = memcmp(*(const void **)v5, (const void *)v22[1], v28);
          v18 = v56;
          v26 = v54;
          v23 = v53;
          if ( !v29 )
            break;
        }
      }
      if ( !*v22 || (v27 = *(_QWORD *)(*v22 + 40LL), v24 = v22, v25 != v27 % v17) )
      {
        v21 = v26;
        v6 = v23;
LABEL_22:
        v57 = v18;
        v31 = (_QWORD *)sub_22077B0(48);
        v32 = v31;
        if ( v31 )
          *v31 = 0;
        v33 = *(_BYTE **)v5;
        v34 = *((_QWORD *)v5 + 1);
        v31[1] = v31 + 3;
        sub_BC50C0(v31 + 1, v33, (__int64)&v33[v34]);
        v8 = v6[1];
        v7 = v51;
        v35 = sub_222DA10(v51, v8, v6[3], 1);
        v36 = v57;
        v37 = v10;
        if ( !v35 )
        {
          v38 = (_QWORD *)*v6;
          v32[5] = v57;
          result = (_QWORD *)((char *)v38 + v21);
          v39 = *(_QWORD **)((char *)v38 + v21);
          if ( v39 )
            goto LABEL_26;
LABEL_41:
          v48 = v6[2];
          v6[2] = v32;
          *v32 = v48;
          if ( v48 )
          {
            v38[*(_QWORD *)(v48 + 40) % v6[1]] = v32;
            result = (_QWORD *)(v21 + *v6);
          }
          *result = v50;
          goto LABEL_27;
        }
        if ( v10 != 1 )
        {
          if ( v10 <= 0xFFFFFFFFFFFFFFFLL )
          {
            v40 = 8 * v10;
            v41 = (void *)sub_22077B0(8 * v10);
            v42 = memset(v41, 0, v40);
            v36 = v57;
            v38 = v42;
            goto LABEL_31;
          }
LABEL_47:
          sub_4261EA(v7, v8, v10, v11);
        }
        v6[6] = 0;
        v38 = v49;
LABEL_31:
        v43 = (_QWORD *)v6[2];
        v6[2] = 0;
        if ( !v43 )
        {
LABEL_38:
          if ( v49 != (void *)*v6 )
          {
            v55 = v38;
            v58 = v36;
            j_j___libc_free_0(*v6, 8LL * v6[1]);
            v38 = v55;
            v36 = v58;
          }
          v6[1] = v37;
          *v6 = v38;
          v32[5] = v36;
          v21 = 8 * (v36 % v37);
          result = (_QWORD *)((char *)v38 + v21);
          v39 = *(_QWORD **)((char *)v38 + v21);
          if ( !v39 )
            goto LABEL_41;
LABEL_26:
          *v32 = *v39;
          result = (_QWORD *)*result;
          *result = v32;
LABEL_27:
          ++v6[3];
          goto LABEL_19;
        }
        v44 = 0;
        while ( 1 )
        {
          v45 = v43;
          v43 = (_QWORD *)*v43;
          v46 = v45[5] % v37;
          v47 = (_QWORD **)&v38[v46];
          if ( *v47 )
            break;
          *v45 = v6[2];
          v6[2] = v45;
          *v47 = v50;
          if ( !*v45 )
          {
            v44 = v46;
LABEL_34:
            if ( !v43 )
              goto LABEL_38;
            continue;
          }
          v38[v44] = v45;
          v44 = v46;
          if ( !v43 )
            goto LABEL_38;
        }
        *v45 = **v47;
        **v47 = v45;
        goto LABEL_34;
      }
      v22 = (_QWORD *)*v22;
    }
    result = (_QWORD *)v26;
    v6 = v23;
    v30 = *v24 == 0;
    v21 = v26;
    if ( v30 )
      goto LABEL_22;
LABEL_19:
    v5 += 32;
  }
  return result;
}
