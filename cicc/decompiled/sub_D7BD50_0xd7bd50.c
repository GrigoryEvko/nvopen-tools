// Function: sub_D7BD50
// Address: 0xd7bd50
//
void __fastcall sub_D7BD50(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  _BYTE *v5; // rax
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // r12
  __int64 v9; // r15
  char **v10; // rsi
  unsigned __int8 v11; // r12
  __int16 v12; // bx
  unsigned __int8 v13; // al
  __int64 v14; // rsi
  __int16 v15; // r9
  unsigned __int16 v16; // ax
  int v17; // r12d
  const void **v18; // rcx
  __int64 v19; // rax
  char *v20; // r15
  char *v21; // r12
  __int64 *v22; // rbx
  __int64 *v23; // r13
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // r12
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // r13
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdi
  __int64 v43; // rbx
  __int64 v44; // r12
  __int64 v45; // rdi
  __int64 v46; // rdi
  char v47; // r12
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  char v52; // r12
  __int64 v53; // r15
  __int16 v54; // ax
  __int64 v55; // rdx
  unsigned __int8 v56; // [rsp-193h] [rbp-193h]
  unsigned __int8 v57; // [rsp-192h] [rbp-192h]
  char v58; // [rsp-191h] [rbp-191h]
  __int16 v59; // [rsp-190h] [rbp-190h]
  unsigned int v60; // [rsp-190h] [rbp-190h]
  __int64 v61; // [rsp-190h] [rbp-190h]
  __int64 v62; // [rsp-190h] [rbp-190h]
  __int16 v63; // [rsp-188h] [rbp-188h]
  char *v64; // [rsp-188h] [rbp-188h]
  _QWORD v65[2]; // [rsp-168h] [rbp-168h] BYREF
  _QWORD v66[2]; // [rsp-158h] [rbp-158h] BYREF
  _QWORD v67[2]; // [rsp-148h] [rbp-148h] BYREF
  __int64 v68; // [rsp-138h] [rbp-138h]
  __int64 v69[2]; // [rsp-128h] [rbp-128h] BYREF
  __int64 v70; // [rsp-118h] [rbp-118h]
  __int64 v71[2]; // [rsp-108h] [rbp-108h] BYREF
  __int64 v72; // [rsp-F8h] [rbp-F8h]
  __int64 v73; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v74; // [rsp-E0h] [rbp-E0h]
  __int64 v75; // [rsp-D8h] [rbp-D8h]
  __int64 v76; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v77; // [rsp-C0h] [rbp-C0h]
  __int64 v78; // [rsp-B8h] [rbp-B8h]
  __int64 v79; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v80; // [rsp-A0h] [rbp-A0h]
  __int64 v81; // [rsp-98h] [rbp-98h]
  __int64 v82; // [rsp-88h] [rbp-88h] BYREF
  __int64 v83; // [rsp-80h] [rbp-80h]
  __int64 v84; // [rsp-78h] [rbp-78h]
  char *v85; // [rsp-68h] [rbp-68h] BYREF
  char *v86; // [rsp-60h] [rbp-60h]
  _QWORD v87[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( (a4 & 6) == 0 )
  {
    **(_BYTE **)a1 = 1;
    v5 = (_BYTE *)sub_BA8B30(*(_QWORD *)(a1 + 8), a2, a3);
    v6 = (__int64)v5;
    if ( v5 )
    {
      v7 = sub_B2FE60(v5);
      v8 = *(_QWORD *)(a1 + 16);
      LOBYTE(v63) = v7;
      LOBYTE(v59) = (*(_BYTE *)(v6 + 33) & 0x40) != 0;
      sub_B2F930(&v85, v6);
      v9 = sub_B2F650((__int64)v85, (__int64)v86);
      if ( v85 != (char *)v87 )
        j_j___libc_free_0(v85, v87[0] + 1LL);
      v82 = v9;
      v10 = (char **)v8;
      sub_D7AC80((__int64)&v85, v8, &v82);
      if ( *(_BYTE *)v6 )
      {
        v85 = (char *)v87;
        v86 = 0;
        v47 = *(_BYTE *)(v6 + 80);
        v48 = sub_22077B0(72);
        v52 = v47 & 1;
        v53 = v48;
        if ( v48 )
        {
          *(_DWORD *)(v48 + 8) = 2;
          *(_QWORD *)(v48 + 16) = 0;
          *(_QWORD *)(v48 + 24) = 0;
          *(_QWORD *)v48 = &unk_49D9770;
          *(_QWORD *)(v48 + 32) = 0;
          v54 = v59 << 8;
          *(_QWORD *)(v53 + 48) = 0;
          LOBYTE(v54) = -57;
          v55 = *(unsigned __int16 *)(v53 + 12);
          LOWORD(v55) = v55 & 0xF800;
          *(_WORD *)(v53 + 12) = v55 | ((v63 << 9) | v54) & 0x7FF;
          *(_QWORD *)(v53 + 40) = v53 + 56;
          if ( (_DWORD)v86 )
          {
            v10 = &v85;
            sub_D76D40(v53 + 40, &v85, v55, v49, v50, v51);
          }
          *(_QWORD *)(v53 + 56) = 0;
          *(_QWORD *)v53 = &unk_49D97D0;
          *(_BYTE *)(v53 + 64) = *(_BYTE *)(v53 + 64) & 0xE0 | (4 * v52);
        }
        if ( v85 != (char *)v87 )
          _libc_free(v85, v10);
        v85 = (char *)v53;
        v46 = *(_QWORD *)(a1 + 24);
      }
      else
      {
        v66[0] = v67;
        v66[1] = 0;
        v65[0] = v66;
        v65[1] = 0;
        v58 = sub_B2D610(v6, 50);
        v57 = sub_B2D610(v6, 51);
        v56 = sub_B2D610(v6, 34);
        v11 = sub_A74710((_QWORD *)(v6 + 120), 0, 22);
        v12 = (unsigned __int8)sub_B2D610(v6, 3);
        v13 = sub_B2D610(v6, 41);
        v14 = 0;
        v67[0] = 0;
        v15 = v13 << 6;
        v67[1] = 0;
        v16 = v59 << 8;
        v68 = 0;
        LOBYTE(v16) = -57;
        v69[0] = 0;
        v69[1] = 0;
        v70 = 0;
        v71[0] = 0;
        v71[1] = 0;
        v60 = ((unsigned __int16)(v63 << 9) | v16) & 0x7C7;
        v72 = 0;
        v73 = 0;
        v74 = 0;
        v75 = 0;
        v76 = 0;
        v77 = 0;
        v78 = 0;
        v79 = 0;
        v82 = 0;
        v83 = 0;
        v80 = 0;
        v17 = v15 & 0x3FF
            | (32 * v12) & 0x3FF
            | (8 * v11) & 0x3FF
            | (4 * v56) & 0x3FF
            | v58 & 0x7F
            | 0x180
            | (2 * v57) & 0x3FF;
        v81 = 0;
        v84 = 0;
        v85 = 0;
        v86 = 0;
        v87[0] = 0;
        v86 = sub_9EB710(0, 0, 0, v18);
        v19 = sub_22077B0(112);
        v64 = (char *)v19;
        if ( v19 )
        {
          v14 = v60;
          sub_9C6E00(v19, v60, 0, v17, (__int64)v65, (__int64)v66, v67, v69, v71, &v73, &v76, &v79, &v82, (__int64)&v85);
        }
        v20 = v86;
        v21 = v85;
        if ( v86 != v85 )
        {
          v61 = v6;
          do
          {
            v22 = (__int64 *)*((_QWORD *)v21 + 12);
            v23 = (__int64 *)*((_QWORD *)v21 + 11);
            if ( v22 != v23 )
            {
              do
              {
                v24 = *v23;
                if ( *v23 )
                {
                  v14 = v23[2] - v24;
                  j_j___libc_free_0(v24, v14);
                }
                v23 += 3;
              }
              while ( v22 != v23 );
              v23 = (__int64 *)*((_QWORD *)v21 + 11);
            }
            if ( v23 )
            {
              v14 = *((_QWORD *)v21 + 13) - (_QWORD)v23;
              j_j___libc_free_0(v23, v14);
            }
            v25 = *((_QWORD *)v21 + 9);
            v26 = *((_QWORD *)v21 + 8);
            if ( v25 != v26 )
            {
              do
              {
                v27 = *(_QWORD *)(v26 + 8);
                if ( v27 != v26 + 24 )
                  _libc_free(v27, v14);
                v26 += 72;
              }
              while ( v25 != v26 );
              v26 = *((_QWORD *)v21 + 8);
            }
            if ( v26 )
            {
              v14 = *((_QWORD *)v21 + 10) - v26;
              j_j___libc_free_0(v26, v14);
            }
            if ( *(char **)v21 != v21 + 24 )
              _libc_free(*(_QWORD *)v21, v14);
            v21 += 112;
          }
          while ( v20 != v21 );
          v6 = v61;
          v21 = v85;
        }
        if ( v21 )
        {
          v14 = v87[0] - (_QWORD)v21;
          j_j___libc_free_0(v21, v87[0] - (_QWORD)v21);
        }
        v28 = v83;
        v29 = v82;
        if ( v83 != v82 )
        {
          do
          {
            v30 = *(_QWORD *)(v29 + 72);
            if ( v30 != v29 + 88 )
              _libc_free(v30, v14);
            v31 = *(_QWORD *)(v29 + 8);
            if ( v31 != v29 + 24 )
              _libc_free(v31, v14);
            v29 += 136;
          }
          while ( v28 != v29 );
          v29 = v82;
        }
        if ( v29 )
        {
          v14 = v84 - v29;
          j_j___libc_free_0(v29, v84 - v29);
        }
        v32 = v80;
        v33 = v79;
        if ( v80 != v79 )
        {
          v62 = v6;
          do
          {
            v34 = *(_QWORD *)(v33 + 48);
            v35 = *(_QWORD *)(v33 + 40);
            if ( v34 != v35 )
            {
              do
              {
                if ( *(_DWORD *)(v35 + 40) > 0x40u )
                {
                  v36 = *(_QWORD *)(v35 + 32);
                  if ( v36 )
                    j_j___libc_free_0_0(v36);
                }
                if ( *(_DWORD *)(v35 + 24) > 0x40u )
                {
                  v37 = *(_QWORD *)(v35 + 16);
                  if ( v37 )
                    j_j___libc_free_0_0(v37);
                }
                v35 += 48;
              }
              while ( v34 != v35 );
              v35 = *(_QWORD *)(v33 + 40);
            }
            if ( v35 )
            {
              v14 = *(_QWORD *)(v33 + 56) - v35;
              j_j___libc_free_0(v35, v14);
            }
            if ( *(_DWORD *)(v33 + 32) > 0x40u )
            {
              v38 = *(_QWORD *)(v33 + 24);
              if ( v38 )
                j_j___libc_free_0_0(v38);
            }
            if ( *(_DWORD *)(v33 + 16) > 0x40u )
            {
              v39 = *(_QWORD *)(v33 + 8);
              if ( v39 )
                j_j___libc_free_0_0(v39);
            }
            v33 += 64;
          }
          while ( v32 != v33 );
          v6 = v62;
          v33 = v79;
        }
        if ( v33 )
        {
          v14 = v81 - v33;
          j_j___libc_free_0(v33, v81 - v33);
        }
        v40 = v77;
        v41 = v76;
        if ( v77 != v76 )
        {
          do
          {
            v42 = *(_QWORD *)(v41 + 16);
            if ( v42 )
            {
              v14 = *(_QWORD *)(v41 + 32) - v42;
              j_j___libc_free_0(v42, v14);
            }
            v41 += 40;
          }
          while ( v40 != v41 );
          v41 = v76;
        }
        if ( v41 )
        {
          v14 = v78 - v41;
          j_j___libc_free_0(v41, v78 - v41);
        }
        v43 = v74;
        v44 = v73;
        if ( v74 != v73 )
        {
          do
          {
            v45 = *(_QWORD *)(v44 + 16);
            if ( v45 )
            {
              v14 = *(_QWORD *)(v44 + 32) - v45;
              j_j___libc_free_0(v45, v14);
            }
            v44 += 40;
          }
          while ( v43 != v44 );
          v44 = v73;
        }
        if ( v44 )
        {
          v14 = v75 - v44;
          j_j___libc_free_0(v44, v75 - v44);
        }
        if ( v71[0] )
        {
          v14 = v72 - v71[0];
          j_j___libc_free_0(v71[0], v72 - v71[0]);
        }
        if ( v69[0] )
        {
          v14 = v70 - v69[0];
          j_j___libc_free_0(v69[0], v70 - v69[0]);
        }
        if ( v67[0] )
        {
          v14 = v68 - v67[0];
          j_j___libc_free_0(v67[0], v68 - v67[0]);
        }
        if ( (_QWORD *)v65[0] != v66 )
          _libc_free(v65[0], v14);
        if ( (_QWORD *)v66[0] != v67 )
          _libc_free(v66[0], v14);
        v46 = *(_QWORD *)(a1 + 24);
        v85 = v64;
      }
      sub_D7A690(v46, v6, (__int64 *)&v85);
      if ( v85 )
        (*(void (__fastcall **)(char *))(*(_QWORD *)v85 + 8LL))(v85);
    }
  }
}
