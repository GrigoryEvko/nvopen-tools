// Function: sub_15D0340
// Address: 0x15d0340
//
void __fastcall sub_15D0340(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r15
  char *v5; // rax
  char *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // rsi
  char *v10; // r14
  __int64 v11; // rax
  __int64 v12; // r13
  bool v13; // bl
  __int64 v14; // r13
  int v15; // ecx
  unsigned __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rax
  int v19; // ecx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r14
  __int64 *v29; // rax
  __int64 v30; // r8
  __int64 *v31; // r13
  __int64 v32; // rdi
  __int64 v33; // r13
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  unsigned __int64 v39; // [rsp+20h] [rbp-80h]
  int v40; // [rsp+28h] [rbp-78h]
  int v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+30h] [rbp-70h] BYREF
  __int64 v44; // [rsp+38h] [rbp-68h] BYREF
  __int64 v45; // [rsp+40h] [rbp-60h] BYREF
  char *v46; // [rsp+48h] [rbp-58h] BYREF
  __int64 v47; // [rsp+50h] [rbp-50h] BYREF
  _BYTE *v48; // [rsp+58h] [rbp-48h]
  _BYTE *v49; // [rsp+60h] [rbp-40h]

  v3 = sub_157EBA0(a2);
  v47 = 0;
  v4 = sub_15F4DF0(v3, 0);
  v5 = *(char **)(a2 + 8);
  v48 = 0;
  v49 = 0;
  v46 = v5;
  sub_15CDD40((__int64 *)&v46);
  v6 = v46;
  if ( v46 )
  {
    v7 = sub_1648700(v46);
LABEL_4:
    v8 = *(_QWORD *)(v7 + 40);
    v9 = v48;
    v43 = v8;
    if ( v48 == v49 )
    {
      sub_1292090((__int64)&v47, v48, &v43);
    }
    else
    {
      if ( v48 )
      {
        *(_QWORD *)v48 = v8;
        v9 = v48;
      }
      v48 = v9 + 8;
    }
    while ( 1 )
    {
      v6 = (char *)*((_QWORD *)v6 + 1);
      if ( !v6 )
        break;
      v7 = sub_1648700(v6);
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 25) <= 9u )
        goto LABEL_4;
    }
  }
  v46 = *(char **)(v4 + 8);
  sub_15CDD40((__int64 *)&v46);
  v10 = v46;
  if ( !v46 )
    goto LABEL_47;
  v11 = sub_1648700(v46);
LABEL_13:
  v12 = *(_QWORD *)(v11 + 40);
  if ( v12 == a2 || (v13 = sub_15CC8F0(a1, v4, *(_QWORD *)(v11 + 40))) || !sub_15CC510(a1, v12) )
  {
    while ( 1 )
    {
      v10 = (char *)*((_QWORD *)v10 + 1);
      if ( !v10 )
        break;
      v11 = sub_1648700(v10);
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 25) <= 9u )
        goto LABEL_13;
    }
LABEL_47:
    v13 = 1;
  }
  v14 = v47;
  v39 = (__int64)&v48[-v47] >> 3;
  if ( !v39 )
    goto LABEL_50;
  v15 = 0;
  v16 = 0;
  while ( 1 )
  {
    v17 = *(_QWORD *)(v14 + 8 * v16);
    v40 = v15;
    v18 = sub_15CC510(a1, v17);
    v19 = v40;
    if ( v18 )
      break;
    v15 = v40 + 1;
    v16 = (unsigned int)(v40 + 1);
    if ( v16 >= v39 )
      goto LABEL_50;
  }
  if ( v17 )
  {
LABEL_21:
    while ( 1 )
    {
      v20 = (unsigned int)(v19 + 1);
      if ( v20 >= v39 )
        break;
      while ( 1 )
      {
        v41 = v20;
        v37 = *(_QWORD *)(v14 + 8 * v20);
        v21 = sub_15CC510(a1, v37);
        v19 = v41;
        v36 = v21;
        if ( !v21 )
          break;
        v22 = *(_QWORD *)(*(_QWORD *)(v17 + 56) + 80LL);
        if ( v22 )
          v22 -= 24;
        if ( v22 == v17 || v37 == v22 )
        {
          v17 = v22;
          goto LABEL_21;
        }
        v23 = sub_15CC510(a1, v17);
        v19 = v41;
        v24 = v36;
        v17 = v23;
        if ( !v23 )
          goto LABEL_21;
        while ( v24 != v17 )
        {
          if ( *(_DWORD *)(v17 + 16) < *(_DWORD *)(v24 + 16) )
          {
            v25 = v17;
            v17 = v24;
            v24 = v25;
          }
          v17 = *(_QWORD *)(v17 + 8);
          if ( !v17 )
            goto LABEL_21;
        }
        v17 = *(_QWORD *)v17;
        v20 = (unsigned int)(v41 + 1);
        if ( v20 >= v39 )
          goto LABEL_34;
      }
    }
LABEL_34:
    v44 = a2;
    v26 = sub_15CC510(a1, v17);
    *(_BYTE *)(a1 + 72) = 0;
    v27 = v26;
    sub_15CC0B0(&v45, a2, v26);
    v46 = (char *)v45;
    sub_15CE4A0(v27 + 24, &v46);
    v28 = v45;
    v45 = 0;
    v29 = sub_15CFF10(a1 + 24, &v44);
    v30 = v29[1];
    v31 = v29;
    v29[1] = v28;
    if ( v30 )
    {
      v32 = *(_QWORD *)(v30 + 24);
      if ( v32 )
      {
        v42 = v30;
        j_j___libc_free_0(v32, *(_QWORD *)(v30 + 40) - v32);
        v30 = v42;
      }
      j_j___libc_free_0(v30, 56);
      v28 = v31[1];
    }
    v33 = v45;
    if ( v45 )
    {
      v34 = *(_QWORD *)(v45 + 24);
      if ( v34 )
        j_j___libc_free_0(v34, *(_QWORD *)(v45 + 40) - v34);
      j_j___libc_free_0(v33, 56);
    }
    if ( v13 )
    {
      v35 = sub_15CC510(a1, v4);
      *(_BYTE *)(a1 + 72) = 0;
      sub_15CE4D0(v35, v28);
    }
    if ( v47 )
      j_j___libc_free_0(v47, &v49[-v47]);
  }
  else
  {
LABEL_50:
    sub_15CE080(&v47);
  }
}
