// Function: sub_AC4A80
// Address: 0xac4a80
//
__int64 __fastcall sub_AC4A80(unsigned __int8 *a1, __int64 a2)
{
  int v2; // eax
  __int64 *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v12; // rax
  const void *v13; // r13
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // rdi
  __int64 v19; // rax
  int v20; // eax
  unsigned __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // r8
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 *v29; // r14
  __int64 v30; // r15
  __int64 v31; // rdx
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 *v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r13
  __int64 v39; // r14
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int8 v48; // al
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 v52; // r15
  __int64 v53; // r14
  __int64 v54; // rax
  unsigned __int8 v55; // r13
  __int16 v56; // bx
  int v57; // eax
  __int64 v58; // r14
  __int64 v59; // rax
  __int16 v60; // bx
  int v61; // eax
  __int64 v62; // [rsp+8h] [rbp-A8h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  __int64 v64; // [rsp+18h] [rbp-98h]
  unsigned __int8 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v67; // [rsp+28h] [rbp-88h]
  __int64 v68; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v69; // [rsp+38h] [rbp-78h]
  char v70; // [rsp+40h] [rbp-70h]
  __int64 v71; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v72; // [rsp+58h] [rbp-58h]
  __int64 v73; // [rsp+60h] [rbp-50h]
  unsigned int v74; // [rsp+68h] [rbp-48h]
  char v75; // [rsp+70h] [rbp-40h]

  v2 = *a1;
  if ( (_BYTE)v2 == 62 )
  {
    v36 = (__int64 *)*((_QWORD *)a1 + 1);
    v37 = *v36;
    v38 = v36[1];
    v39 = v36[2];
    v40 = sub_BD2C40(24, unk_3F28984);
    v7 = v40;
    if ( !v40 )
      return 0;
    sub_BD35F0(v40, *(_QWORD *)(v37 + 8), 5);
    v41 = *(_DWORD *)(v7 + 4);
    *(_WORD *)(v7 + 2) = 62;
    v17 = *(_QWORD *)(v7 - 96) == 0;
    *(_DWORD *)(v7 + 4) = v41 & 0x38000000 | 3;
    if ( !v17 )
    {
      v42 = *(_QWORD *)(v7 - 88);
      **(_QWORD **)(v7 - 80) = v42;
      if ( v42 )
        *(_QWORD *)(v42 + 16) = *(_QWORD *)(v7 - 80);
    }
    *(_QWORD *)(v7 - 96) = v37;
    v43 = *(_QWORD *)(v37 + 16);
    *(_QWORD *)(v7 - 88) = v43;
    if ( v43 )
      *(_QWORD *)(v43 + 16) = v7 - 88;
    *(_QWORD *)(v7 - 80) = v37 + 16;
    *(_QWORD *)(v37 + 16) = v7 - 96;
    if ( *(_QWORD *)(v7 - 64) )
    {
      v44 = *(_QWORD *)(v7 - 56);
      **(_QWORD **)(v7 - 48) = v44;
      if ( v44 )
        *(_QWORD *)(v44 + 16) = *(_QWORD *)(v7 - 48);
    }
    *(_QWORD *)(v7 - 64) = v38;
    if ( v38 )
    {
      v45 = *(_QWORD *)(v38 + 16);
      *(_QWORD *)(v7 - 56) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = v7 - 56;
      *(_QWORD *)(v7 - 48) = v38 + 16;
      *(_QWORD *)(v38 + 16) = v7 - 64;
    }
    if ( *(_QWORD *)(v7 - 32) )
    {
      v46 = *(_QWORD *)(v7 - 24);
      **(_QWORD **)(v7 - 16) = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = *(_QWORD *)(v7 - 16);
    }
    *(_QWORD *)(v7 - 32) = v39;
    if ( v39 )
    {
      v47 = *(_QWORD *)(v39 + 16);
      *(_QWORD *)(v7 - 24) = v47;
      if ( v47 )
        *(_QWORD *)(v47 + 16) = v7 - 24;
      *(_QWORD *)(v7 - 16) = v39 + 16;
      *(_QWORD *)(v39 + 16) = v7 - 32;
    }
  }
  else if ( (unsigned __int8)v2 > 0x3Eu )
  {
    if ( (_BYTE)v2 != 63 )
    {
LABEL_58:
      if ( (unsigned int)(v2 - 13) > 0x11 )
        BUG();
      v51 = (__int64 *)*((_QWORD *)a1 + 1);
      v52 = *v51;
      v53 = v51[1];
      v54 = sub_BD2C40(24, unk_3F2898C);
      v7 = v54;
      if ( v54 )
      {
        v55 = a1[1];
        v56 = *a1;
        sub_BD35F0(v54, *(_QWORD *)(v52 + 8), 5);
        v57 = *(_DWORD *)(v7 + 4);
        *(_WORD *)(v7 + 2) = v56;
        *(_DWORD *)(v7 + 4) = v57 & 0x38000000 | 2;
        sub_AC2B30(v7 - 64, v52);
        sub_AC2B30(v7 - 32, v53);
        *(_BYTE *)(v7 + 1) = (2 * v55) | *(_BYTE *)(v7 + 1) & 1;
      }
      return v7;
    }
    v12 = (__int64 *)*((_QWORD *)a1 + 1);
    v13 = (const void *)*((_QWORD *)a1 + 3);
    v14 = *((_QWORD *)a1 + 4);
    v15 = *v12;
    v64 = v12[1];
    v7 = sub_BD2C40(64, unk_3F28980);
    if ( v7 )
    {
      v16 = *(_QWORD *)(v15 + 8);
      v17 = *(_BYTE *)(v16 + 8) == 18;
      v18 = *(_QWORD *)(v16 + 24);
      LODWORD(v71) = v14;
      BYTE4(v71) = v17;
      v19 = sub_BCE1B0(v18, v71);
      sub_BD35F0(v7, v19, 5);
      v20 = *(_DWORD *)(v7 + 4);
      *(_WORD *)(v7 + 2) = 63;
      *(_QWORD *)(v7 + 24) = v7 + 40;
      *(_QWORD *)(v7 + 32) = 0x400000000LL;
      *(_DWORD *)(v7 + 4) = v20 & 0x38000000 | 2;
      v21 = (4 * v14) >> 2;
      sub_AC2B30(v7 - 64, v15);
      sub_AC2B30(v7 - 32, v64);
      v24 = *(unsigned int *)(v7 + 36);
      v25 = 0;
      LODWORD(v26) = 0;
      *(_DWORD *)(v7 + 32) = 0;
      v27 = v7 + 40;
      if ( v21 > v24 )
      {
        sub_C8D5F0(v7 + 24, v7 + 40, (4 * v14) >> 2, 4);
        v26 = *(unsigned int *)(v7 + 32);
        v25 = 4 * v26;
      }
      if ( 4 * v14 )
      {
        memcpy((void *)(*(_QWORD *)(v7 + 24) + v25), v13, 4 * v14);
        LODWORD(v26) = *(_DWORD *)(v7 + 32);
      }
      v28 = *(_QWORD *)(v7 + 8);
      *(_DWORD *)(v7 + 32) = v26 + v21;
      *(_QWORD *)(v7 + 56) = sub_B4E660(v13, v14, v28, v22, v23, v27);
    }
  }
  else
  {
    if ( (_BYTE)v2 != 34 )
    {
      if ( (_BYTE)v2 == 61 )
      {
        v4 = (__int64 *)*((_QWORD *)a1 + 1);
        v5 = *v4;
        v6 = v4[1];
        v7 = sub_BD2C40(24, unk_3F28988);
        if ( v7 )
        {
          sub_BD35F0(v7, *(_QWORD *)(*(_QWORD *)(v5 + 8) + 24LL), 5);
          v8 = *(_DWORD *)(v7 + 4);
          *(_WORD *)(v7 + 2) = 61;
          *(_DWORD *)(v7 + 4) = v8 & 0x38000000 | 2;
          sub_AC2B30(v7 - 64, v5);
          if ( *(_QWORD *)(v7 - 32) )
          {
            v9 = *(_QWORD *)(v7 - 24);
            **(_QWORD **)(v7 - 16) = v9;
            if ( v9 )
              *(_QWORD *)(v9 + 16) = *(_QWORD *)(v7 - 16);
          }
          *(_QWORD *)(v7 - 32) = v6;
          if ( v6 )
          {
            v10 = *(_QWORD *)(v6 + 16);
            *(_QWORD *)(v7 - 24) = v10;
            if ( v10 )
              *(_QWORD *)(v10 + 16) = v7 - 24;
            *(_QWORD *)(v7 - 16) = v6 + 16;
            *(_QWORD *)(v6 + 16) = v7 - 32;
          }
          return v7;
        }
        return 0;
      }
      if ( (unsigned int)(v2 - 38) <= 0xC )
      {
        v58 = **((_QWORD **)a1 + 1);
        v59 = sub_BD2C40(24, unk_3F28990);
        v7 = v59;
        if ( v59 )
        {
          v60 = *a1;
          sub_BD35F0(v59, a2, 5);
          v61 = *(_DWORD *)(v7 + 4);
          *(_WORD *)(v7 + 2) = v60;
          *(_DWORD *)(v7 + 4) = v61 & 0x38000000 | 1;
          sub_AC2B30(v7 - 32, v58);
        }
        return v7;
      }
      goto LABEL_58;
    }
    v17 = a1[80] == 0;
    v70 = 0;
    if ( v17 )
    {
      v29 = (__int64 *)*((_QWORD *)a1 + 1);
      v30 = *((_QWORD *)a1 + 5);
      v31 = *v29;
      v65 = a1[1];
      v32 = v29 + 1;
      v33 = *((_QWORD *)a1 + 2);
      v75 = 0;
      v34 = (unsigned int)v33;
      v35 = v33 - 1;
    }
    else
    {
      v67 = *((_DWORD *)a1 + 14);
      if ( v67 > 0x40 )
        sub_C43780(&v66, a1 + 48);
      else
        v66 = *((_QWORD *)a1 + 6);
      v69 = *((_DWORD *)a1 + 18);
      if ( v69 > 0x40 )
        sub_C43780(&v68, a1 + 64);
      else
        v68 = *((_QWORD *)a1 + 8);
      v48 = a1[1];
      v49 = *((_QWORD *)a1 + 1);
      v70 = 1;
      v30 = *((_QWORD *)a1 + 5);
      v65 = v48;
      v50 = *((_QWORD *)a1 + 2);
      v32 = (__int64 *)(v49 + 8);
      v31 = *(v32 - 1);
      v75 = 1;
      v34 = (unsigned int)v50;
      v35 = v50 - 1;
      LODWORD(v50) = v67;
      v67 = 0;
      v72 = v50;
      v71 = v66;
      LODWORD(v50) = v69;
      v69 = 0;
      v74 = v50;
      v73 = v68;
    }
    v62 = v35;
    v63 = v31;
    v7 = sub_BD2C40(80, v34);
    if ( v7 )
      sub_AC4880(v7, v30, v63, v32, v62, a2, (__int64)&v71, v34 & 0x7FFFFFF);
    if ( v75 )
    {
      v75 = 0;
      if ( v74 > 0x40 && v73 )
        j_j___libc_free_0_0(v73);
      if ( v72 > 0x40 && v71 )
        j_j___libc_free_0_0(v71);
    }
    *(_BYTE *)(v7 + 1) = (2 * v65) | *(_BYTE *)(v7 + 1) & 1;
    if ( v70 )
    {
      v70 = 0;
      if ( v69 > 0x40 && v68 )
        j_j___libc_free_0_0(v68);
      if ( v67 > 0x40 && v66 )
        j_j___libc_free_0_0(v66);
    }
  }
  return v7;
}
