// Function: sub_35D9270
// Address: 0x35d9270
//
unsigned __int64 *__fastcall sub_35D9270(unsigned __int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r14
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // r15
  __int64 v15; // r11
  int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // r15
  unsigned __int64 v21; // r10
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // eax
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rax
  unsigned int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // eax
  unsigned __int64 i; // r13
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  unsigned __int8 *v41; // rsi
  int v42; // eax
  __int64 v43; // [rsp+0h] [rbp-60h]
  __int64 *v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  unsigned __int64 v46; // [rsp+10h] [rbp-50h]
  unsigned __int64 v48; // [rsp+20h] [rbp-40h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]
  unsigned __int64 v50; // [rsp+20h] [rbp-40h]
  __int64 v51; // [rsp+20h] [rbp-40h]
  __int64 v52; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v7 = a1[1];
  v8 = *a1;
  v9 = 0x4EC4EC4EC4EC4EC5LL * ((__int64)(v7 - *a1) >> 3);
  if ( v9 == 0x13B13B13B13B13BLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0x4EC4EC4EC4EC4EC5LL * ((__int64)(v7 - v8) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 + v9;
  v13 = a2 - v8;
  if ( v11 )
  {
    v39 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_52:
    v43 = a4;
    v44 = a3;
    v40 = sub_22077B0(v39);
    v13 = a2 - v8;
    a3 = v44;
    v52 = v40;
    a4 = v43;
    v46 = v40 + v39;
    v14 = v40 + 104;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x13B13B13B13B13BLL )
      v12 = 0x13B13B13B13B13BLL;
    v39 = 104 * v12;
    goto LABEL_52;
  }
  v46 = 0;
  v14 = 104;
  v52 = 0;
LABEL_7:
  v15 = v52 + v13;
  if ( v15 )
  {
    v16 = *((_DWORD *)a3 + 2);
    v17 = a3[4];
    *((_DWORD *)a3 + 2) = 0;
    *(_BYTE *)(v15 + 96) = 0;
    *(_DWORD *)(v15 + 8) = v16;
    v18 = *a3;
    *(_QWORD *)(v15 + 32) = v17;
    *(_QWORD *)v15 = v18;
    LODWORD(v18) = *((_DWORD *)a3 + 6);
    *((_DWORD *)a3 + 6) = 0;
    v19 = *(_BYTE *)(a4 + 40) == 0;
    *(_DWORD *)(v15 + 24) = v18;
    *(_QWORD *)(v15 + 16) = a3[2];
    *(_QWORD *)(v15 + 40) = a3[5];
    *(_WORD *)(v15 + 48) = *((_WORD *)a3 + 24);
    *(_DWORD *)(v15 + 56) = *(_DWORD *)a4;
    *(_DWORD *)(v15 + 60) = *(_DWORD *)(a4 + 4);
    *(_QWORD *)(v15 + 64) = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(v15 + 72) = *(_QWORD *)(a4 + 16);
    if ( !v19 )
    {
      v41 = *(unsigned __int8 **)(a4 + 24);
      *(_QWORD *)(v15 + 80) = v41;
      if ( v41 )
      {
        v45 = v15;
        v51 = a4;
        sub_B976B0(a4 + 24, v41, v15 + 80);
        a4 = v51;
        v15 = v45;
        *(_QWORD *)(v51 + 24) = 0;
      }
      v42 = *(_DWORD *)(a4 + 32);
      *(_BYTE *)(v15 + 96) = 1;
      *(_DWORD *)(v15 + 88) = v42;
    }
  }
  if ( a2 != v8 )
  {
    v20 = v52;
    v21 = v8;
    while ( 1 )
    {
      if ( !v20 )
      {
LABEL_14:
        v21 += 104LL;
        v24 = v20 + 104;
        if ( a2 == v21 )
          goto LABEL_23;
        goto LABEL_15;
      }
      v25 = *(_DWORD *)(v21 + 8);
      *(_DWORD *)(v20 + 8) = v25;
      if ( v25 <= 0x40 )
      {
        *(_QWORD *)v20 = *(_QWORD *)v21;
        v22 = *(_DWORD *)(v21 + 24);
        *(_DWORD *)(v20 + 24) = v22;
        if ( v22 <= 0x40 )
          goto LABEL_12;
      }
      else
      {
        v48 = v21;
        sub_C43780(v20, (const void **)v21);
        v21 = v48;
        v26 = *(_DWORD *)(v48 + 24);
        *(_DWORD *)(v20 + 24) = v26;
        if ( v26 <= 0x40 )
        {
LABEL_12:
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v21 + 16);
          goto LABEL_13;
        }
      }
      v49 = v21;
      sub_C43780(v20 + 16, (const void **)(v21 + 16));
      v21 = v49;
LABEL_13:
      *(_QWORD *)(v20 + 32) = *(_QWORD *)(v21 + 32);
      *(_QWORD *)(v20 + 40) = *(_QWORD *)(v21 + 40);
      *(_BYTE *)(v20 + 48) = *(_BYTE *)(v21 + 48);
      *(_BYTE *)(v20 + 49) = *(_BYTE *)(v21 + 49);
      *(_DWORD *)(v20 + 56) = *(_DWORD *)(v21 + 56);
      *(_DWORD *)(v20 + 60) = *(_DWORD *)(v21 + 60);
      *(_QWORD *)(v20 + 64) = *(_QWORD *)(v21 + 64);
      v23 = *(_QWORD *)(v21 + 72);
      *(_BYTE *)(v20 + 96) = 0;
      *(_QWORD *)(v20 + 72) = v23;
      if ( !*(_BYTE *)(v21 + 96) )
        goto LABEL_14;
      v27 = *(_QWORD *)(v21 + 80);
      *(_QWORD *)(v20 + 80) = v27;
      if ( v27 )
      {
        v50 = v21;
        sub_B96E90(v20 + 80, v27, 1);
        v21 = v50;
      }
      v28 = *(_DWORD *)(v21 + 88);
      v21 += 104LL;
      *(_BYTE *)(v20 + 96) = 1;
      *(_DWORD *)(v20 + 88) = v28;
      v24 = v20 + 104;
      if ( a2 == v21 )
      {
LABEL_23:
        v14 = v20 + 208;
        break;
      }
LABEL_15:
      v20 = v24;
    }
  }
  if ( a2 != v7 )
  {
    while ( 1 )
    {
      v31 = *(_DWORD *)(v5 + 8);
      *(_DWORD *)(v14 + 8) = v31;
      if ( v31 <= 0x40 )
      {
        *(_QWORD *)v14 = *(_QWORD *)v5;
        v29 = *(_DWORD *)(v5 + 24);
        *(_DWORD *)(v14 + 24) = v29;
        if ( v29 > 0x40 )
          goto LABEL_32;
      }
      else
      {
        sub_C43780(v14, (const void **)v5);
        v32 = *(_DWORD *)(v5 + 24);
        *(_DWORD *)(v14 + 24) = v32;
        if ( v32 > 0x40 )
        {
LABEL_32:
          sub_C43780(v14 + 16, (const void **)(v5 + 16));
          goto LABEL_28;
        }
      }
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v5 + 16);
LABEL_28:
      v30 = *(_QWORD *)(v5 + 32);
      v19 = *(_BYTE *)(v5 + 96) == 0;
      *(_BYTE *)(v14 + 96) = 0;
      *(_QWORD *)(v14 + 32) = v30;
      *(_QWORD *)(v14 + 40) = *(_QWORD *)(v5 + 40);
      *(_BYTE *)(v14 + 48) = *(_BYTE *)(v5 + 48);
      *(_BYTE *)(v14 + 49) = *(_BYTE *)(v5 + 49);
      *(_DWORD *)(v14 + 56) = *(_DWORD *)(v5 + 56);
      *(_DWORD *)(v14 + 60) = *(_DWORD *)(v5 + 60);
      *(_QWORD *)(v14 + 64) = *(_QWORD *)(v5 + 64);
      *(_QWORD *)(v14 + 72) = *(_QWORD *)(v5 + 72);
      if ( v19 )
      {
        v5 += 104;
        v14 += 104;
        if ( v7 == v5 )
          break;
      }
      else
      {
        v33 = *(_QWORD *)(v5 + 80);
        *(_QWORD *)(v14 + 80) = v33;
        if ( v33 )
          sub_B96E90(v14 + 80, v33, 1);
        v34 = *(_DWORD *)(v5 + 88);
        v5 += 104;
        *(_BYTE *)(v14 + 96) = 1;
        v14 += 104;
        *(_DWORD *)(v14 - 16) = v34;
        if ( v7 == v5 )
          break;
      }
    }
  }
  for ( i = v8; i != v7; i += 104LL )
  {
    if ( *(_BYTE *)(i + 96) )
    {
      v37 = *(_QWORD *)(i + 80);
      *(_BYTE *)(i + 96) = 0;
      if ( v37 )
        sub_B91220(i + 80, v37);
    }
    if ( *(_DWORD *)(i + 24) > 0x40u )
    {
      v36 = *(_QWORD *)(i + 16);
      if ( v36 )
        j_j___libc_free_0_0(v36);
    }
    if ( *(_DWORD *)(i + 8) > 0x40u && *(_QWORD *)i )
      j_j___libc_free_0_0(*(_QWORD *)i);
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  *a1 = v52;
  a1[1] = v14;
  a1[2] = v46;
  return a1;
}
