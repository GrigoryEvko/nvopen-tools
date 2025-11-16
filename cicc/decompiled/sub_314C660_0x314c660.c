// Function: sub_314C660
// Address: 0x314c660
//
__int64 __fastcall sub_314C660(__int64 a1, __int64 a2, char a3)
{
  int v5; // eax
  bool v6; // zf
  __int64 v7; // r12
  char *v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // r9
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // ecx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  int v23; // r8d
  _BYTE *v24; // r9
  unsigned int v25; // ecx
  int v26; // r8d
  unsigned int v27; // eax
  unsigned int v28; // r10d
  __int64 v29; // rax
  __int64 v30; // r11
  __int64 v31; // rsi
  unsigned __int64 *v32; // rax
  char *v33; // rsi
  unsigned __int64 *v34; // rdx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  int v38; // r8d
  _BYTE *v39; // r9
  unsigned int v40; // ecx
  int v41; // r8d
  unsigned int v42; // eax
  unsigned int v43; // r10d
  __int64 v44; // rax
  __int64 v45; // r11
  __int64 v46; // rsi
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // rdi
  int v50; // r8d
  _BYTE *v51; // r9
  unsigned int v52; // ecx
  int v53; // r8d
  unsigned int v54; // eax
  unsigned int v55; // r10d
  __int64 v56; // rax
  __int64 v57; // r11
  __int64 v58; // rsi
  __int64 v60; // rcx
  __int64 v61; // rcx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rsi
  unsigned __int64 v64; // rdi
  int v65; // r8d
  _BYTE *v66; // r9
  unsigned int v67; // ecx
  int v68; // r8d
  unsigned int v69; // eax
  unsigned int v70; // r10d
  __int64 v71; // rax
  __int64 v72; // r11
  __int64 v73; // rsi
  __int64 v74; // rcx
  __int64 v75; // rcx
  char *v76; // [rsp+0h] [rbp-A0h]
  __int64 v77; // [rsp+8h] [rbp-98h]
  __int64 v78; // [rsp+10h] [rbp-90h]
  unsigned int v79; // [rsp+10h] [rbp-90h]
  unsigned int v80; // [rsp+10h] [rbp-90h]
  unsigned int v81; // [rsp+10h] [rbp-90h]
  unsigned int v82; // [rsp+10h] [rbp-90h]
  unsigned int v83; // [rsp+10h] [rbp-90h]
  unsigned int v84; // [rsp+10h] [rbp-90h]
  unsigned int v85; // [rsp+10h] [rbp-90h]
  char v86; // [rsp+10h] [rbp-90h]
  unsigned int v87; // [rsp+10h] [rbp-90h]
  char v88; // [rsp+10h] [rbp-90h]
  char v89; // [rsp+10h] [rbp-90h]
  char v90; // [rsp+10h] [rbp-90h]
  __int64 v91; // [rsp+20h] [rbp-80h]
  __int64 v92; // [rsp+28h] [rbp-78h]
  _QWORD *v93; // [rsp+30h] [rbp-70h] BYREF
  int v94; // [rsp+38h] [rbp-68h]
  _QWORD v95[2]; // [rsp+40h] [rbp-60h] BYREF
  char *v96; // [rsp+50h] [rbp-50h]
  size_t v97; // [rsp+58h] [rbp-48h]
  _OWORD v98[4]; // [rsp+60h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v5 = *(_DWORD *)(a2 - 24);
  else
    v5 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_BYTE *)(a1 + 16) = 0;
  if ( v5 > 0 )
  {
    v6 = a3 == 0;
    v7 = 8;
    v8 = " ";
    v91 = a2 - 16;
    if ( !v6 )
      v8 = "\n";
    v76 = v8;
    v92 = 16LL * ((unsigned int)(v5 - 1) >> 1) + 24;
    while ( 1 )
    {
      v9 = *(_BYTE *)(a2 - 16);
      if ( (v9 & 2) != 0 )
        v10 = *(_QWORD *)(a2 - 32);
      else
        v10 = v91 - 8LL * ((v9 >> 2) & 0xF);
      v11 = sub_B91420(*(_QWORD *)(v10 + v7 - 8));
      v12 = *(_BYTE *)(a2 - 16);
      v14 = v13;
      if ( (v12 & 2) != 0 )
        v15 = *(_QWORD *)(a2 - 32);
      else
        v15 = v91 - 8LL * ((v12 >> 2) & 0xF);
      v16 = *(_QWORD *)(*(_QWORD *)(v15 + v7) + 136LL);
      v17 = *(_QWORD *)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v17 = **(_QWORD **)(v16 + 24);
      v18 = *(_QWORD *)(a1 + 8);
      v19 = v17;
      if ( v18 )
      {
        if ( v18 == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v77 = v14;
        v78 = v11;
        sub_2241490((unsigned __int64 *)a1, v76, 1u);
        v19 = v17;
        v11 = v78;
        v14 = v77;
      }
      if ( v14 == 15 )
      {
        if ( *(_QWORD *)v11 != 0x6576726573657270LL
          || *(_DWORD *)(v11 + 8) != 1683975775
          || *(_WORD *)(v11 + 12) != 29793
          || *(_BYTE *)(v11 + 14) != 97 )
        {
LABEL_136:
          BUG();
        }
        if ( v19 <= 9 )
        {
          v88 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 1u, 0);
          v39 = v93;
          LOBYTE(v40) = v88;
          goto LABEL_57;
        }
        if ( v19 <= 0x63 )
        {
          v87 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 2u, 0);
          v39 = v93;
          v40 = v87;
        }
        else
        {
          if ( v19 <= 0x3E7 )
          {
            v36 = 3;
          }
          else
          {
            v35 = (unsigned int)v17;
            if ( v19 <= 0x270F )
            {
              v36 = 4;
            }
            else
            {
              LODWORD(v36) = 1;
              while ( 1 )
              {
                v37 = v35;
                v38 = v36;
                v36 = (unsigned int)(v36 + 4);
                v35 /= 0x2710u;
                if ( v37 <= 0x1869F )
                  break;
                if ( (unsigned int)v35 <= 0x63 )
                {
                  v80 = v19;
                  v93 = v95;
                  v36 = (unsigned int)(v38 + 5);
                  goto LABEL_54;
                }
                if ( (unsigned int)v35 <= 0x3E7 )
                {
                  v36 = (unsigned int)(v38 + 6);
                  break;
                }
                if ( (unsigned int)v35 <= 0x270F )
                {
                  v36 = (unsigned int)(v38 + 7);
                  break;
                }
              }
            }
          }
          v80 = v19;
          v93 = v95;
LABEL_54:
          sub_2240A50((__int64 *)&v93, v36, 0);
          v39 = v93;
          v40 = v80;
          v41 = v94 - 1;
          do
          {
            v42 = v40 % 0x64;
            v43 = v40;
            v40 /= 0x64u;
            v44 = 2 * v42;
            v45 = (unsigned int)(v44 + 1);
            LOBYTE(v44) = a00010203040506[v44];
            v39[v41] = a00010203040506[v45];
            v46 = (unsigned int)(v41 - 1);
            v41 -= 2;
            v39[v46] = v44;
          }
          while ( v43 > 0x270F );
          if ( v43 <= 0x3E7 )
          {
LABEL_57:
            *v39 = v40 + 48;
            goto LABEL_58;
          }
        }
        v75 = 2 * v40;
        v39[1] = a00010203040506[(unsigned int)(v75 + 1)];
        *v39 = a00010203040506[v75];
LABEL_58:
        v32 = sub_2241130((unsigned __int64 *)&v93, 0, 0, ".abi_preserve ", 0xEu);
        v96 = (char *)v98;
        v33 = (char *)*v32;
        v34 = v32 + 2;
        if ( (unsigned __int64 *)*v32 == v32 + 2 )
          goto LABEL_108;
        goto LABEL_77;
      }
      if ( v14 != 18 )
      {
        if ( v14 != 16 || *(_QWORD *)v11 ^ 0x6576726573657270LL | *(_QWORD *)(v11 + 8) ^ 0x72657466615F6E5FLL )
          goto LABEL_136;
        if ( v19 <= 9 )
        {
          v89 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 1u, 0);
          v51 = v93;
          LOBYTE(v52) = v89;
          goto LABEL_75;
        }
        if ( v19 <= 0x63 )
        {
          v82 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 2u, 0);
          v51 = v93;
          v52 = v82;
        }
        else
        {
          if ( v19 <= 0x3E7 )
          {
            v48 = 3;
          }
          else
          {
            v47 = (unsigned int)v17;
            if ( v19 <= 0x270F )
            {
              v48 = 4;
            }
            else
            {
              LODWORD(v48) = 1;
              while ( 1 )
              {
                v49 = v47;
                v50 = v48;
                v48 = (unsigned int)(v48 + 4);
                v47 /= 0x2710u;
                if ( v49 <= 0x1869F )
                  break;
                if ( (unsigned int)v47 <= 0x63 )
                {
                  v81 = v19;
                  v93 = v95;
                  v48 = (unsigned int)(v50 + 5);
                  goto LABEL_72;
                }
                if ( (unsigned int)v47 <= 0x3E7 )
                {
                  v48 = (unsigned int)(v50 + 6);
                  break;
                }
                if ( (unsigned int)v47 <= 0x270F )
                {
                  v48 = (unsigned int)(v50 + 7);
                  break;
                }
              }
            }
          }
          v81 = v19;
          v93 = v95;
LABEL_72:
          sub_2240A50((__int64 *)&v93, v48, 0);
          v51 = v93;
          v52 = v81;
          v53 = v94 - 1;
          do
          {
            v54 = v52 % 0x64;
            v55 = v52;
            v52 /= 0x64u;
            v56 = 2 * v54;
            v57 = (unsigned int)(v56 + 1);
            LOBYTE(v56) = a00010203040506[v56];
            v51[v53] = a00010203040506[v57];
            v58 = (unsigned int)(v53 - 1);
            v53 -= 2;
            v51[v58] = v56;
          }
          while ( v55 > 0x270F );
          if ( v55 <= 0x3E7 )
          {
LABEL_75:
            *v51 = v52 + 48;
            goto LABEL_76;
          }
        }
        v60 = 2 * v52;
        v51[1] = a00010203040506[(unsigned int)(v60 + 1)];
        *v51 = a00010203040506[v60];
LABEL_76:
        v32 = sub_2241130((unsigned __int64 *)&v93, 0, 0, ".abi_preserve_after ", 0x14u);
        v96 = (char *)v98;
        v33 = (char *)*v32;
        v34 = v32 + 2;
        if ( (unsigned __int64 *)*v32 == v32 + 2 )
          goto LABEL_108;
        goto LABEL_77;
      }
      if ( *(_QWORD *)v11 ^ 0x6576726573657270LL | *(_QWORD *)(v11 + 8) ^ 0x72746E6F635F6E5FLL
        || *(_WORD *)(v11 + 16) != 27759 )
      {
        if ( *(_QWORD *)v11 ^ 0x6576726573657270LL | *(_QWORD *)(v11 + 8) ^ 0x6F66696E755F6E5FLL
          || *(_WORD *)(v11 + 16) != 28018 )
        {
          goto LABEL_136;
        }
        if ( v19 <= 9 )
        {
          v86 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 1u, 0);
          v24 = v93;
          LOBYTE(v25) = v86;
          goto LABEL_35;
        }
        if ( v19 <= 0x63 )
        {
          v83 = v19;
          v93 = v95;
          sub_2240A50((__int64 *)&v93, 2u, 0);
          v24 = v93;
          v25 = v83;
        }
        else
        {
          if ( v19 <= 0x3E7 )
          {
            v21 = 3;
          }
          else
          {
            v20 = (unsigned int)v17;
            if ( v19 <= 0x270F )
            {
              v21 = 4;
            }
            else
            {
              LODWORD(v21) = 1;
              while ( 1 )
              {
                v22 = v20;
                v23 = v21;
                v21 = (unsigned int)(v21 + 4);
                v20 /= 0x2710u;
                if ( v22 <= 0x1869F )
                  break;
                if ( (unsigned int)v20 <= 0x63 )
                {
                  v79 = v19;
                  v93 = v95;
                  v21 = (unsigned int)(v23 + 5);
                  goto LABEL_32;
                }
                if ( (unsigned int)v20 <= 0x3E7 )
                {
                  v21 = (unsigned int)(v23 + 6);
                  break;
                }
                if ( (unsigned int)v20 <= 0x270F )
                {
                  v21 = (unsigned int)(v23 + 7);
                  break;
                }
              }
            }
          }
          v79 = v19;
          v93 = v95;
LABEL_32:
          sub_2240A50((__int64 *)&v93, v21, 0);
          v24 = v93;
          v25 = v79;
          v26 = v94 - 1;
          do
          {
            v27 = v25 % 0x64;
            v28 = v25;
            v25 /= 0x64u;
            v29 = 2 * v27;
            v30 = (unsigned int)(v29 + 1);
            LOBYTE(v29) = a00010203040506[v29];
            v24[v26] = a00010203040506[v30];
            v31 = (unsigned int)(v26 - 1);
            v26 -= 2;
            v24[v31] = v29;
          }
          while ( v28 > 0x270F );
          if ( v28 <= 0x3E7 )
          {
LABEL_35:
            *v24 = v25 + 48;
            goto LABEL_36;
          }
        }
        v61 = 2 * v25;
        v24[1] = a00010203040506[(unsigned int)(v61 + 1)];
        *v24 = a00010203040506[v61];
LABEL_36:
        v32 = sub_2241130((unsigned __int64 *)&v93, 0, 0, ".abi_preserve_uniform ", 0x16u);
        v96 = (char *)v98;
        v33 = (char *)*v32;
        v34 = v32 + 2;
        if ( (unsigned __int64 *)*v32 == v32 + 2 )
          goto LABEL_108;
        goto LABEL_77;
      }
      if ( v19 <= 9 )
        break;
      if ( v19 <= 0x63 )
      {
        v85 = v19;
        v93 = v95;
        sub_2240A50((__int64 *)&v93, 2u, 0);
        v66 = v93;
        v67 = v85;
      }
      else
      {
        if ( v19 <= 0x3E7 )
        {
          v63 = 3;
        }
        else
        {
          v62 = (unsigned int)v17;
          if ( v19 <= 0x270F )
          {
            v63 = 4;
          }
          else
          {
            LODWORD(v63) = 1;
            while ( 1 )
            {
              v64 = v62;
              v65 = v63;
              v63 = (unsigned int)(v63 + 4);
              v62 /= 0x2710u;
              if ( v64 <= 0x1869F )
                break;
              if ( (unsigned int)v62 <= 0x63 )
              {
                v84 = v19;
                v93 = v95;
                v63 = (unsigned int)(v65 + 5);
                goto LABEL_103;
              }
              if ( (unsigned int)v62 <= 0x3E7 )
              {
                v63 = (unsigned int)(v65 + 6);
                break;
              }
              if ( (unsigned int)v62 <= 0x270F )
              {
                v63 = (unsigned int)(v65 + 7);
                break;
              }
            }
          }
        }
        v84 = v19;
        v93 = v95;
LABEL_103:
        sub_2240A50((__int64 *)&v93, v63, 0);
        v66 = v93;
        v67 = v84;
        v68 = v94 - 1;
        do
        {
          v69 = v67 % 0x64;
          v70 = v67;
          v67 /= 0x64u;
          v71 = 2 * v69;
          v72 = (unsigned int)(v71 + 1);
          LOBYTE(v71) = a00010203040506[v71];
          v66[v68] = a00010203040506[v72];
          v73 = (unsigned int)(v68 - 1);
          v68 -= 2;
          v66[v73] = v71;
        }
        while ( v70 > 0x270F );
        if ( v70 <= 0x3E7 )
          goto LABEL_106;
      }
      v74 = 2 * v67;
      v66[1] = a00010203040506[(unsigned int)(v74 + 1)];
      *v66 = a00010203040506[v74];
LABEL_107:
      v32 = sub_2241130((unsigned __int64 *)&v93, 0, 0, ".abi_preserve_control ", 0x16u);
      v96 = (char *)v98;
      v33 = (char *)*v32;
      v34 = v32 + 2;
      if ( (unsigned __int64 *)*v32 == v32 + 2 )
      {
LABEL_108:
        v98[0] = _mm_loadu_si128((const __m128i *)v32 + 1);
        goto LABEL_78;
      }
LABEL_77:
      v96 = v33;
      *(_QWORD *)&v98[0] = v32[2];
LABEL_78:
      v97 = v32[1];
      *v32 = (unsigned __int64)v34;
      v32[1] = 0;
      *((_BYTE *)v32 + 16) = 0;
      sub_2241490((unsigned __int64 *)a1, v96, v97);
      if ( v96 != (char *)v98 )
        j_j___libc_free_0((unsigned __int64)v96);
      if ( v93 != v95 )
        j_j___libc_free_0((unsigned __int64)v93);
      v7 += 16;
      if ( v92 == v7 )
        return a1;
    }
    v90 = v19;
    v93 = v95;
    sub_2240A50((__int64 *)&v93, 1u, 0);
    v66 = v93;
    LOBYTE(v67) = v90;
LABEL_106:
    *v66 = v67 + 48;
    goto LABEL_107;
  }
  return a1;
}
