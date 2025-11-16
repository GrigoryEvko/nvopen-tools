// Function: sub_3937240
// Address: 0x3937240
//
__int64 __fastcall sub_3937240(__int64 a1, __int64 a2, char a3)
{
  int v4; // eax
  char *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned int v15; // ecx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  int v19; // r8d
  _BYTE *v20; // r9
  unsigned int v21; // ecx
  int v22; // esi
  unsigned int v23; // eax
  unsigned int v24; // r10d
  __int64 v25; // rax
  __int64 v26; // r11
  __int64 v27; // rdx
  unsigned __int64 *v28; // rax
  char *v29; // rsi
  unsigned __int64 *v30; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rdi
  int v36; // r8d
  _BYTE *v37; // r9
  unsigned int v38; // ecx
  int v39; // esi
  unsigned int v40; // eax
  unsigned int v41; // r10d
  __int64 v42; // rax
  __int64 v43; // r11
  __int64 v44; // rdx
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rdi
  int v48; // r8d
  _BYTE *v49; // r9
  unsigned int v50; // ecx
  int v51; // esi
  unsigned int v52; // eax
  unsigned int v53; // r10d
  __int64 v54; // rax
  __int64 v55; // r11
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // rsi
  unsigned __int64 v61; // rdi
  int v62; // r8d
  _BYTE *v63; // r9
  unsigned int v64; // ecx
  int v65; // esi
  unsigned int v66; // eax
  unsigned int v67; // r10d
  __int64 v68; // rax
  __int64 v69; // r11
  __int64 v70; // rdx
  __int64 v71; // rcx
  char *v72; // [rsp+8h] [rbp-98h]
  __int64 v73; // [rsp+10h] [rbp-90h]
  __int64 v74; // [rsp+18h] [rbp-88h]
  unsigned int v75; // [rsp+18h] [rbp-88h]
  unsigned int v76; // [rsp+18h] [rbp-88h]
  unsigned int v77; // [rsp+18h] [rbp-88h]
  unsigned int v78; // [rsp+18h] [rbp-88h]
  unsigned int v79; // [rsp+18h] [rbp-88h]
  unsigned int v80; // [rsp+18h] [rbp-88h]
  unsigned int v81; // [rsp+18h] [rbp-88h]
  unsigned int v82; // [rsp+18h] [rbp-88h]
  char v83; // [rsp+18h] [rbp-88h]
  char v84; // [rsp+18h] [rbp-88h]
  char v85; // [rsp+18h] [rbp-88h]
  char v86; // [rsp+18h] [rbp-88h]
  int v87; // [rsp+24h] [rbp-7Ch]
  _QWORD *v88; // [rsp+30h] [rbp-70h] BYREF
  int v89; // [rsp+38h] [rbp-68h]
  _QWORD v90[2]; // [rsp+40h] [rbp-60h] BYREF
  char *v91; // [rsp+50h] [rbp-50h]
  size_t v92; // [rsp+58h] [rbp-48h]
  _OWORD v93[4]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v87 = v4;
  *(_QWORD *)a1 = a1 + 16;
  if ( v4 > 0 )
  {
    v6 = " ";
    if ( a3 )
      v6 = "\n";
    v7 = 0;
    v72 = v6;
    while ( 1 )
    {
      v8 = sub_161E970(*(_QWORD *)(a2 + 8 * (v7 - *(unsigned int *)(a2 + 8))));
      v10 = v9;
      v11 = v8;
      v12 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v7 + 1 - *(unsigned int *)(a2 + 8))) + 136LL);
      if ( *(_DWORD *)(v12 + 32) <= 0x40u )
        v13 = *(_QWORD *)(v12 + 24);
      else
        v13 = **(_QWORD **)(v12 + 24);
      v14 = *(_QWORD *)(a1 + 8);
      v15 = v13;
      if ( v14 )
      {
        if ( v14 == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v73 = v9;
        v74 = v11;
        sub_2241490((unsigned __int64 *)a1, v72, 1u);
        v15 = v13;
        v11 = v74;
        v10 = v73;
      }
      if ( v10 == 15 )
      {
        if ( v15 <= 9 )
        {
          v84 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 1u, 0);
          v49 = v88;
          LOBYTE(v50) = v84;
          goto LABEL_69;
        }
        if ( v15 <= 0x63 )
        {
          v79 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 2u, 0);
          v49 = v88;
          v50 = v79;
        }
        else
        {
          if ( v15 <= 0x3E7 )
          {
            v46 = 3;
          }
          else
          {
            v45 = (unsigned int)v13;
            if ( v15 <= 0x270F )
            {
              v46 = 4;
            }
            else
            {
              LODWORD(v46) = 1;
              while ( 1 )
              {
                v47 = v45;
                v48 = v46;
                v46 = (unsigned int)(v46 + 4);
                v45 /= 0x2710u;
                if ( v47 <= 0x1869F )
                  break;
                if ( (unsigned int)v45 <= 0x63 )
                {
                  v78 = v15;
                  v88 = v90;
                  v46 = (unsigned int)(v48 + 5);
                  goto LABEL_66;
                }
                if ( (unsigned int)v45 <= 0x3E7 )
                {
                  v46 = (unsigned int)(v48 + 6);
                  break;
                }
                if ( (unsigned int)v45 <= 0x270F )
                {
                  v46 = (unsigned int)(v48 + 7);
                  break;
                }
              }
            }
          }
          v78 = v15;
          v88 = v90;
LABEL_66:
          sub_2240A50((__int64 *)&v88, v46, 0);
          v49 = v88;
          v50 = v78;
          v51 = v89 - 1;
          do
          {
            v52 = v50 % 0x64;
            v53 = v50;
            v50 /= 0x64u;
            v54 = 2 * v52;
            v55 = (unsigned int)(v54 + 1);
            LOBYTE(v54) = a00010203040506[v54];
            v49[v51] = a00010203040506[v55];
            v56 = (unsigned int)(v51 - 1);
            v51 -= 2;
            v49[v56] = v54;
          }
          while ( v53 > 0x270F );
          if ( v53 <= 0x3E7 )
          {
LABEL_69:
            *v49 = v50 + 48;
            goto LABEL_70;
          }
        }
        v57 = 2 * v50;
        v49[1] = a00010203040506[(unsigned int)(v57 + 1)];
        *v49 = a00010203040506[v57];
LABEL_70:
        v28 = sub_2241130((unsigned __int64 *)&v88, 0, 0, ".abi_preserve ", 0xEu);
        v91 = (char *)v93;
        v29 = (char *)*v28;
        v30 = v28 + 2;
        if ( (unsigned __int64 *)*v28 == v28 + 2 )
          goto LABEL_94;
        goto LABEL_27;
      }
      if ( v10 != 18 )
      {
        if ( v15 <= 9 )
        {
          v83 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 1u, 0);
          v20 = v88;
          LOBYTE(v21) = v83;
          goto LABEL_25;
        }
        if ( v15 <= 0x63 )
        {
          v76 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 2u, 0);
          v20 = v88;
          v21 = v76;
        }
        else
        {
          if ( v15 <= 0x3E7 )
          {
            v17 = 3;
          }
          else
          {
            v16 = (unsigned int)v13;
            if ( v15 <= 0x270F )
            {
              v17 = 4;
            }
            else
            {
              LODWORD(v17) = 1;
              while ( 1 )
              {
                v18 = v16;
                v19 = v17;
                v17 = (unsigned int)(v17 + 4);
                v16 /= 0x2710u;
                if ( v18 <= 0x1869F )
                  break;
                if ( (unsigned int)v16 <= 0x63 )
                {
                  v75 = v15;
                  v88 = v90;
                  v17 = (unsigned int)(v19 + 5);
                  goto LABEL_22;
                }
                if ( (unsigned int)v16 <= 0x3E7 )
                {
                  v17 = (unsigned int)(v19 + 6);
                  break;
                }
                if ( (unsigned int)v16 <= 0x270F )
                {
                  v17 = (unsigned int)(v19 + 7);
                  break;
                }
              }
            }
          }
          v75 = v15;
          v88 = v90;
LABEL_22:
          sub_2240A50((__int64 *)&v88, v17, 0);
          v20 = v88;
          v21 = v75;
          v22 = v89 - 1;
          do
          {
            v23 = v21 % 0x64;
            v24 = v21;
            v21 /= 0x64u;
            v25 = 2 * v23;
            v26 = (unsigned int)(v25 + 1);
            LOBYTE(v25) = a00010203040506[v25];
            v20[v22] = a00010203040506[v26];
            v27 = (unsigned int)(v22 - 1);
            v22 -= 2;
            v20[v27] = v25;
          }
          while ( v24 > 0x270F );
          if ( v24 <= 0x3E7 )
          {
LABEL_25:
            *v20 = v21 + 48;
            goto LABEL_26;
          }
        }
        v32 = 2 * v21;
        v20[1] = a00010203040506[(unsigned int)(v32 + 1)];
        *v20 = a00010203040506[v32];
LABEL_26:
        v28 = sub_2241130((unsigned __int64 *)&v88, 0, 0, ".abi_preserve_after ", 0x14u);
        v91 = (char *)v93;
        v29 = (char *)*v28;
        v30 = v28 + 2;
        if ( (unsigned __int64 *)*v28 == v28 + 2 )
          goto LABEL_94;
        goto LABEL_27;
      }
      if ( *(_QWORD *)v11 ^ 0x6576726573657270LL | *(_QWORD *)(v11 + 8) ^ 0x72746E6F635F6E5FLL
        || *(_WORD *)(v11 + 16) != 27759 )
      {
        if ( v15 <= 9 )
        {
          v86 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 1u, 0);
          v37 = v88;
          LOBYTE(v38) = v86;
          goto LABEL_52;
        }
        if ( v15 <= 0x63 )
        {
          v80 = v15;
          v88 = v90;
          sub_2240A50((__int64 *)&v88, 2u, 0);
          v37 = v88;
          v38 = v80;
        }
        else
        {
          if ( v15 <= 0x3E7 )
          {
            v34 = 3;
          }
          else
          {
            v33 = (unsigned int)v13;
            if ( v15 <= 0x270F )
            {
              v34 = 4;
            }
            else
            {
              LODWORD(v34) = 1;
              while ( 1 )
              {
                v35 = v33;
                v36 = v34;
                v34 = (unsigned int)(v34 + 4);
                v33 /= 0x2710u;
                if ( v35 <= 0x1869F )
                  break;
                if ( (unsigned int)v33 <= 0x63 )
                {
                  v77 = v15;
                  v88 = v90;
                  v34 = (unsigned int)(v36 + 5);
                  goto LABEL_49;
                }
                if ( (unsigned int)v33 <= 0x3E7 )
                {
                  v34 = (unsigned int)(v36 + 6);
                  break;
                }
                if ( (unsigned int)v33 <= 0x270F )
                {
                  v34 = (unsigned int)(v36 + 7);
                  break;
                }
              }
            }
          }
          v77 = v15;
          v88 = v90;
LABEL_49:
          sub_2240A50((__int64 *)&v88, v34, 0);
          v37 = v88;
          v38 = v77;
          v39 = v89 - 1;
          do
          {
            v40 = v38 % 0x64;
            v41 = v38;
            v38 /= 0x64u;
            v42 = 2 * v40;
            v43 = (unsigned int)(v42 + 1);
            LOBYTE(v42) = a00010203040506[v42];
            v37[v39] = a00010203040506[v43];
            v44 = (unsigned int)(v39 - 1);
            v39 -= 2;
            v37[v44] = v42;
          }
          while ( v41 > 0x270F );
          if ( v41 <= 0x3E7 )
          {
LABEL_52:
            *v37 = v38 + 48;
            goto LABEL_53;
          }
        }
        v58 = 2 * v38;
        v37[1] = a00010203040506[(unsigned int)(v58 + 1)];
        *v37 = a00010203040506[v58];
LABEL_53:
        v28 = sub_2241130((unsigned __int64 *)&v88, 0, 0, ".abi_preserve_uniform ", 0x16u);
        v91 = (char *)v93;
        v29 = (char *)*v28;
        v30 = v28 + 2;
        if ( (unsigned __int64 *)*v28 == v28 + 2 )
          goto LABEL_94;
        goto LABEL_27;
      }
      if ( v15 <= 9 )
        break;
      if ( v15 <= 0x63 )
      {
        v82 = v15;
        v88 = v90;
        sub_2240A50((__int64 *)&v88, 2u, 0);
        v63 = v88;
        v64 = v82;
      }
      else
      {
        if ( v15 <= 0x3E7 )
        {
          v60 = 3;
        }
        else
        {
          v59 = (unsigned int)v13;
          if ( v15 <= 0x270F )
          {
            v60 = 4;
          }
          else
          {
            LODWORD(v60) = 1;
            while ( 1 )
            {
              v61 = v59;
              v62 = v60;
              v60 = (unsigned int)(v60 + 4);
              v59 /= 0x2710u;
              if ( v61 <= 0x1869F )
                break;
              if ( (unsigned int)v59 <= 0x63 )
              {
                v81 = v15;
                v88 = v90;
                v60 = (unsigned int)(v62 + 5);
                goto LABEL_89;
              }
              if ( (unsigned int)v59 <= 0x3E7 )
              {
                v60 = (unsigned int)(v62 + 6);
                break;
              }
              if ( (unsigned int)v59 <= 0x270F )
              {
                v60 = (unsigned int)(v62 + 7);
                break;
              }
            }
          }
        }
        v81 = v15;
        v88 = v90;
LABEL_89:
        sub_2240A50((__int64 *)&v88, v60, 0);
        v63 = v88;
        v64 = v81;
        v65 = v89 - 1;
        do
        {
          v66 = v64 % 0x64;
          v67 = v64;
          v64 /= 0x64u;
          v68 = 2 * v66;
          v69 = (unsigned int)(v68 + 1);
          LOBYTE(v68) = a00010203040506[v68];
          v63[v65] = a00010203040506[v69];
          v70 = (unsigned int)(v65 - 1);
          v65 -= 2;
          v63[v70] = v68;
        }
        while ( v67 > 0x270F );
        if ( v67 <= 0x3E7 )
          goto LABEL_92;
      }
      v71 = 2 * v64;
      v63[1] = a00010203040506[(unsigned int)(v71 + 1)];
      *v63 = a00010203040506[v71];
LABEL_93:
      v28 = sub_2241130((unsigned __int64 *)&v88, 0, 0, ".abi_preserve_control ", 0x16u);
      v91 = (char *)v93;
      v29 = (char *)*v28;
      v30 = v28 + 2;
      if ( (unsigned __int64 *)*v28 == v28 + 2 )
      {
LABEL_94:
        v93[0] = _mm_loadu_si128((const __m128i *)v28 + 1);
        goto LABEL_28;
      }
LABEL_27:
      v91 = v29;
      *(_QWORD *)&v93[0] = v28[2];
LABEL_28:
      v92 = v28[1];
      *v28 = (unsigned __int64)v30;
      v28[1] = 0;
      *((_BYTE *)v28 + 16) = 0;
      sub_2241490((unsigned __int64 *)a1, v91, v92);
      if ( v91 != (char *)v93 )
        j_j___libc_free_0((unsigned __int64)v91);
      if ( v88 != v90 )
        j_j___libc_free_0((unsigned __int64)v88);
      v7 += 2;
      if ( v87 <= (int)v7 )
        return a1;
    }
    v85 = v15;
    v88 = v90;
    sub_2240A50((__int64 *)&v88, 1u, 0);
    v63 = v88;
    LOBYTE(v64) = v85;
LABEL_92:
    *v63 = v64 + 48;
    goto LABEL_93;
  }
  return a1;
}
