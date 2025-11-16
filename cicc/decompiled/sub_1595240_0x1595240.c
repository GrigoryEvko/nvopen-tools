// Function: sub_1595240
// Address: 0x1595240
//
__int64 __fastcall sub_1595240(unsigned __int8 *a1, __int64 a2)
{
  int v2; // ecx
  __int64 *v3; // rdx
  _QWORD *v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int8 v8; // r12
  __int16 v9; // bx
  int v10; // eax
  __int64 v12; // rbx
  __int64 *v13; // rcx
  __int64 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 v19; // rax
  __int16 v20; // bx
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r13
  _QWORD *v24; // r12
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned int v27; // eax
  int v28; // eax
  _QWORD *v29; // r12
  __int64 v30; // r13
  __int64 v31; // r14
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // r14
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rax
  _QWORD *v40; // r13
  int v41; // eax
  size_t v42; // r14
  void *v43; // rdi
  __int64 v44; // rbx
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // rax
  _QWORD *v48; // r13
  int v49; // eax
  size_t v50; // r14
  void *v51; // rdi
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 v54; // rax
  __int16 v55; // bx
  int v56; // eax
  __int64 v57; // rax
  __int64 *src; // [rsp+0h] [rbp-40h]
  void *srca; // [rsp+0h] [rbp-40h]
  void *v60; // [rsp+8h] [rbp-38h]
  void *v61; // [rsp+8h] [rbp-38h]
  void *v62; // [rsp+8h] [rbp-38h]

  v2 = *a1;
  v3 = (__int64 *)*((_QWORD *)a1 + 1);
  switch ( (char)v2 )
  {
    case ' ':
      v12 = *((_QWORD *)a1 + 5);
      v13 = v3 + 1;
      v8 = a1[1];
      v14 = (__int64 *)*v3;
      v15 = *((_QWORD *)a1 + 2) - 1LL;
      if ( !v12 )
      {
        v57 = *v14;
        if ( *(_BYTE *)(*v14 + 8) == 16 )
          v57 = **(_QWORD **)(v57 + 16);
        v12 = *(_QWORD *)(v57 + 24);
      }
      src = v13;
      v60 = v14;
      v16 = sub_1648A60(40, *((unsigned int *)a1 + 4));
      v7 = v16;
      if ( v16 )
        sub_1595090(v16, v12, (__int64)v60, src, v15, a2);
      goto LABEL_5;
    case '3':
      v17 = *v3;
      v18 = v3[1];
      v19 = sub_1648A60(32, 2);
      v7 = v19;
      if ( !v19 )
        return v7;
      v20 = *((_WORD *)a1 + 1);
      sub_1648CB0(v19, a2, 5);
      v21 = *(_DWORD *)(v7 + 20);
      *(_WORD *)(v7 + 18) = 51;
      *(_DWORD *)(v7 + 20) = v21 & 0xF0000000 | 2;
      goto LABEL_12;
    case '4':
      v17 = *v3;
      v18 = v3[1];
      v22 = sub_1648A60(32, 2);
      v7 = v22;
      if ( !v22 )
        return v7;
      v20 = *((_WORD *)a1 + 1);
      sub_1648CB0(v22, a2, 5);
      *(_DWORD *)(v7 + 20) = *(_DWORD *)(v7 + 20) & 0xF0000000 | 2;
      *(_WORD *)(v7 + 18) = 52;
LABEL_12:
      *(_WORD *)(v7 + 24) = v20;
      goto LABEL_13;
    case '7':
      v23 = *v3;
      v24 = (_QWORD *)v3[1];
      v25 = v3[2];
      v26 = sub_1648A60(24, 3);
      v7 = v26;
      if ( v26 )
      {
        sub_1648CB0(v26, *v24, 5);
        v27 = *(_DWORD *)(v7 + 20) & 0xF0000000;
        *(_WORD *)(v7 + 18) = 55;
        *(_DWORD *)(v7 + 20) = v27 | 3;
        sub_1593B40((_QWORD *)(v7 - 72), v23);
        sub_1593B40((_QWORD *)(v7 - 48), (__int64)v24);
        sub_1593B40((_QWORD *)(v7 - 24), v25);
      }
      return v7;
    case ';':
      v17 = *v3;
      v18 = v3[1];
      v7 = sub_1648A60(24, 2);
      if ( !v7 )
        return v7;
      sub_1648CB0(v7, *(_QWORD *)(*(_QWORD *)v17 + 24LL), 5);
      v28 = *(_DWORD *)(v7 + 20);
      *(_WORD *)(v7 + 18) = 59;
      *(_DWORD *)(v7 + 20) = v28 & 0xF0000000 | 2;
      goto LABEL_13;
    case '<':
      v29 = (_QWORD *)*v3;
      v30 = v3[1];
      v31 = v3[2];
      v32 = sub_1648A60(24, 3);
      v7 = v32;
      if ( v32 )
      {
        sub_1648CB0(v32, *v29, 5);
        v33 = *(_DWORD *)(v7 + 20) & 0xF0000000;
        *(_WORD *)(v7 + 18) = 60;
        *(_DWORD *)(v7 + 20) = v33 | 3;
        sub_1593B40((_QWORD *)(v7 - 72), (__int64)v29);
        sub_1593B40((_QWORD *)(v7 - 48), v30);
        sub_1593B40((_QWORD *)(v7 - 24), v31);
      }
      return v7;
    case '=':
      v34 = *v3;
      v17 = v3[1];
      v18 = v3[2];
      v7 = sub_1648A60(24, 3);
      if ( !v7 )
        return v7;
      v35 = sub_16463B0(*(_QWORD *)(*(_QWORD *)v34 + 24LL), *(_QWORD *)(*(_QWORD *)v18 + 32LL));
      sub_1648CB0(v7, v35, 5);
      v36 = *(_DWORD *)(v7 + 20);
      *(_WORD *)(v7 + 18) = 61;
      *(_DWORD *)(v7 + 20) = v36 & 0xF0000000 | 3;
      sub_1593B40((_QWORD *)(v7 - 72), v34);
LABEL_13:
      sub_1593B40((_QWORD *)(v7 - 48), v17);
      sub_1593B40((_QWORD *)(v7 - 24), v18);
      return v7;
    case '>':
      v37 = *((_QWORD *)a1 + 4);
      v38 = *v3;
      v61 = (void *)*((_QWORD *)a1 + 3);
      v39 = sub_1648A60(56, 1);
      v7 = v39;
      if ( !v39 )
        return v7;
      v40 = (_QWORD *)(v39 - 24);
      sub_1648CB0(v39, a2, 5);
      v41 = *(_DWORD *)(v7 + 20);
      v42 = 4 * v37;
      v43 = (void *)(v7 + 40);
      *(_WORD *)(v7 + 18) = 62;
      *(_QWORD *)(v7 + 24) = v7 + 40;
      v44 = (4 * v37) >> 2;
      *(_DWORD *)(v7 + 20) = v41 & 0xF0000000 | 1;
      *(_QWORD *)(v7 + 32) = 0x400000000LL;
      if ( v42 > 0x10 )
      {
        sub_16CD150(v7 + 24, v7 + 40, v44, 4);
        v43 = (void *)(*(_QWORD *)(v7 + 24) + 4LL * *(unsigned int *)(v7 + 32));
      }
      else if ( !v42 )
      {
        goto LABEL_27;
      }
      memcpy(v43, v61, v42);
      LODWORD(v42) = *(_DWORD *)(v7 + 32);
LABEL_27:
      *(_DWORD *)(v7 + 32) = v42 + v44;
      sub_1593B40(v40, v38);
      return v7;
    case '?':
      v45 = *((_QWORD *)a1 + 4);
      v46 = v3[1];
      v62 = (void *)*v3;
      srca = (void *)*((_QWORD *)a1 + 3);
      v47 = sub_1648A60(56, 2);
      v7 = v47;
      if ( !v47 )
        return v7;
      v48 = (_QWORD *)(v47 - 48);
      sub_1648CB0(v47, a2, 5);
      v49 = *(_DWORD *)(v7 + 20);
      v50 = 4 * v45;
      v51 = (void *)(v7 + 40);
      *(_WORD *)(v7 + 18) = 63;
      *(_QWORD *)(v7 + 24) = v7 + 40;
      v52 = (4 * v45) >> 2;
      *(_DWORD *)(v7 + 20) = v49 & 0xF0000000 | 2;
      *(_QWORD *)(v7 + 32) = 0x400000000LL;
      if ( v50 > 0x10 )
      {
        sub_16CD150(v7 + 24, v7 + 40, v52, 4);
        v51 = (void *)(*(_QWORD *)(v7 + 24) + 4LL * *(unsigned int *)(v7 + 32));
      }
      else if ( !v50 )
      {
        goto LABEL_31;
      }
      memcpy(v51, srca, v50);
      LODWORD(v50) = *(_DWORD *)(v7 + 32);
LABEL_31:
      *(_DWORD *)(v7 + 32) = v50 + v52;
      sub_1593B40(v48, (__int64)v62);
      sub_1593B40((_QWORD *)(v7 - 24), v46);
      return v7;
    default:
      if ( (unsigned int)(v2 - 36) <= 0xC )
      {
        v53 = *v3;
        v54 = sub_1648A60(24, 1);
        v7 = v54;
        if ( v54 )
        {
          v55 = *a1;
          sub_1648CB0(v54, a2, 5);
          v56 = *(_DWORD *)(v7 + 20);
          *(_WORD *)(v7 + 18) = v55;
          *(_DWORD *)(v7 + 20) = v56 & 0xF0000000 | 1;
          sub_1593B40((_QWORD *)(v7 - 24), v53);
        }
      }
      else
      {
        v4 = (_QWORD *)*v3;
        v5 = v3[1];
        v6 = sub_1648A60(24, 2);
        v7 = v6;
        if ( v6 )
        {
          v8 = a1[1];
          v9 = *a1;
          sub_1648CB0(v6, *v4, 5);
          v10 = *(_DWORD *)(v7 + 20);
          *(_WORD *)(v7 + 18) = v9;
          *(_DWORD *)(v7 + 20) = v10 & 0xF0000000 | 2;
          sub_1593B40((_QWORD *)(v7 - 48), (__int64)v4);
          sub_1593B40((_QWORD *)(v7 - 24), v5);
LABEL_5:
          *(_BYTE *)(v7 + 17) = (2 * v8) | *(_BYTE *)(v7 + 17) & 1;
        }
      }
      return v7;
  }
}
