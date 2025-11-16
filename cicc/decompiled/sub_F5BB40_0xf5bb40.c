// Function: sub_F5BB40
// Address: 0xf5bb40
//
__int64 __fastcall sub_F5BB40(unsigned __int64 a1, unsigned __int8 a2, unsigned int a3, __int64 a4)
{
  char v4; // r11
  unsigned __int8 v7; // bl
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v11; // r15
  int v12; // edx
  __int64 v13; // rsi
  __int64 *v14; // rax
  unsigned __int8 v15; // r9
  char v16; // r11
  __int64 *v17; // rdi
  _BYTE *v18; // r8
  unsigned int v19; // eax
  unsigned int v20; // r10d
  _BYTE *v21; // r8
  unsigned __int8 v22; // r9
  char v23; // r11
  unsigned __int64 v24; // rax
  unsigned int v25; // ecx
  unsigned int v26; // eax
  __int64 v27; // rax
  unsigned int v28; // ebx
  __int64 v29; // rdi
  __int64 v30; // r13
  __int64 *v31; // rax
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // r11
  _QWORD *v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rbx
  __int64 v43; // rax
  unsigned int v44; // r12d
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  int v55; // edx
  int v56; // eax
  __int64 v57; // rax
  char v58; // [rsp+Eh] [rbp-E2h]
  char v59; // [rsp+Eh] [rbp-E2h]
  unsigned __int8 v60; // [rsp+Fh] [rbp-E1h]
  __int64 *v61; // [rsp+10h] [rbp-E0h]
  char v62; // [rsp+18h] [rbp-D8h]
  _BYTE *v63; // [rsp+18h] [rbp-D8h]
  __int64 v64; // [rsp+18h] [rbp-D8h]
  unsigned int v65; // [rsp+18h] [rbp-D8h]
  unsigned int v67; // [rsp+20h] [rbp-D0h]
  __int64 v68; // [rsp+20h] [rbp-D0h]
  _BYTE *v69; // [rsp+20h] [rbp-D0h]
  char v71; // [rsp+37h] [rbp-B9h] BYREF
  __int64 v72; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v73; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v74; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v75; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v76; // [rsp+58h] [rbp-98h]
  _QWORD v77[4]; // [rsp+60h] [rbp-90h] BYREF
  char v78; // [rsp+80h] [rbp-70h]
  char v79; // [rsp+81h] [rbp-6Fh]
  __int64 v80; // [rsp+90h] [rbp-60h] BYREF
  int v81; // [rsp+98h] [rbp-58h] BYREF
  __int64 v82; // [rsp+A0h] [rbp-50h]
  int *v83; // [rsp+A8h] [rbp-48h]
  int *v84; // [rsp+B0h] [rbp-40h]
  __int64 v85; // [rsp+B8h] [rbp-38h]

  v4 = a3;
  v7 = a3;
  if ( *(_BYTE *)a1 != 58 )
  {
    if ( *(_BYTE *)a1 != 85 )
      return 0;
    v8 = *(_QWORD *)(a1 - 32);
    if ( !v8 )
      return 0;
    if ( (*(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a1 + 80) || *(_DWORD *)(v8 + 36) != 180)
      && (*(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a1 + 80) || *(_DWORD *)(v8 + 36) != 181)
      && (*(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a1 + 80) || *(_DWORD *)(v8 + 36) != 15) )
    {
      return 0;
    }
  }
  v9 = a3;
  LOBYTE(v9) = a2 | a3;
  if ( a2 | (unsigned __int8)a3 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v12 = *(unsigned __int8 *)(v11 + 8);
    if ( (unsigned int)(v12 - 17) <= 1 )
      LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
    v62 = v4;
    v9 = 0;
    if ( (_BYTE)v12 == 12 && (unsigned int)sub_BCB060(v11) <= 0x80 )
    {
      v13 = v7;
      v71 = 0;
      v81 = 0;
      v82 = 0;
      v83 = &v81;
      v84 = &v81;
      v85 = 0;
      v14 = sub_F5A610(a1, v7, &v80, 0, &v71);
      v15 = a2;
      v16 = v62;
      v9 = *((unsigned __int8 *)v14 + 64);
      v61 = v14;
      if ( !(_BYTE)v9 )
        goto LABEL_21;
      v72 = v11;
      v17 = (__int64 *)v11;
      v18 = (_BYTE *)v14[1];
      v13 = v14[2];
      if ( v18[v13 - 1] == 0xFF )
      {
        while ( v13 )
        {
          if ( v18[v13 - 1] != 0xFF )
          {
            v59 = v62;
            v64 = v14[1];
            v53 = (_QWORD *)sub_BD5C60(a1);
            v54 = sub_BCD140(v53, v13);
            v18 = (_BYTE *)v64;
            v15 = a2;
            v72 = v54;
            v17 = (__int64 *)v54;
            v16 = v59;
            v55 = *(unsigned __int8 *)(v11 + 8);
            if ( (unsigned int)(v55 - 17) <= 1 )
            {
              v56 = *(_DWORD *)(v11 + 32);
              BYTE4(v74) = (_BYTE)v55 == 18;
              LODWORD(v74) = v56;
              v13 = v74;
              v57 = sub_BCE1B0(v17, v74);
              v16 = v59;
              v15 = a2;
              v72 = v57;
              v18 = (_BYTE *)v64;
              v17 = (__int64 *)v57;
            }
            goto LABEL_20;
          }
          --v13;
        }
        goto LABEL_21;
      }
LABEL_20:
      v58 = v16;
      v60 = v15;
      v63 = v18;
      v67 = sub_BCB060((__int64)v17);
      v19 = sub_BCB060(v11);
      v20 = v67;
      if ( v19 < v67 )
      {
LABEL_21:
        v9 = 0;
LABEL_22:
        sub_F4F5A0(v82, v13);
        return v9;
      }
      v76 = v67;
      v21 = v63;
      v22 = v60;
      v23 = v58;
      if ( v67 > 0x40 )
      {
        v13 = -1;
        v65 = v67;
        v69 = v21;
        sub_C43690((__int64)&v75, -1, 1);
        v23 = v58;
        v22 = v60;
        v20 = v65;
        v21 = v69;
      }
      else
      {
        v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v67;
        if ( !v67 )
          v24 = 0;
        v75 = v24;
      }
      if ( a2 )
      {
        v22 = (v20 & 0xF) == 0;
        if ( !v20 )
          goto LABEL_83;
        if ( !v7 && (v20 & 0xF) != 0 )
          goto LABEL_41;
      }
      else
      {
        v23 = v9;
        if ( !v20 )
        {
LABEL_54:
          v28 = 14;
          goto LABEL_55;
        }
      }
      v25 = 0;
      v13 = v20 - 1;
      while ( 1 )
      {
        v26 = (char)*v21;
        if ( *v21 == 0xFF )
        {
          v27 = ~(1LL << v25);
          if ( v76 > 0x40 )
            *(_QWORD *)(v75 + 8LL * (v25 >> 6)) &= v27;
          else
            v75 &= v27;
        }
        else
        {
          if ( (((unsigned __int8)v25 ^ (unsigned __int8)v26) & 7) != 0 )
            v22 = 0;
          else
            v22 &= (v20 >> 3) - 1 - (v25 >> 3) == v26 >> 3;
          v23 &= (_DWORD)v13 - v25 == v26;
        }
        if ( v20 <= ++v25 )
          break;
        ++v21;
        if ( !v22 && !v23 )
          goto LABEL_41;
      }
      if ( !v22 )
      {
        if ( v23 )
          goto LABEL_54;
LABEL_41:
        v9 = 0;
LABEL_42:
        if ( v76 > 0x40 && v75 )
          j_j___libc_free_0_0(v75);
        goto LABEL_22;
      }
LABEL_83:
      v28 = 15;
LABEL_55:
      v29 = a1;
      v30 = a1 + 24;
      v31 = (__int64 *)sub_B43CA0(v29);
      v32 = sub_B6E160(v31, v28, (__int64)&v72, 1);
      v33 = *v61;
      v73 = v33;
      if ( v72 != *(_QWORD *)(v33 + 8) )
      {
        v79 = 1;
        v77[0] = "trunc";
        v78 = 3;
        v34 = sub_B522D0(v33, v72, 0, (__int64)v77, v30, 0);
        v37 = *(unsigned int *)(a4 + 8);
        if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v37 + 1, 8u, v35, v36);
          v37 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v37) = v34;
        ++*(_DWORD *)(a4 + 8);
        v73 = v34;
      }
      v79 = 1;
      v38 = 0;
      v77[0] = "rev";
      v78 = 3;
      if ( v32 )
        v38 = *(_QWORD *)(v32 + 24);
      v13 = 2;
      v68 = v38;
      v39 = sub_BD2C40(88, 2u);
      v42 = (__int64)v39;
      if ( v39 )
      {
        sub_B44260((__int64)v39, **(_QWORD **)(v68 + 16), 56, 2u, v30, 0);
        *(_QWORD *)(v42 + 72) = 0;
        v13 = v68;
        sub_B4A290(v42, v68, v32, &v73, 1, (__int64)v77, 0, 0);
      }
      v43 = *(unsigned int *)(a4 + 8);
      if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        v13 = a4 + 16;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v43 + 1, 8u, v40, v41);
        v43 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v43) = v42;
      v44 = v76;
      ++*(_DWORD *)(a4 + 8);
      if ( !v44 )
        goto LABEL_71;
      if ( v44 <= 0x40 )
      {
        if ( v75 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v44) )
          goto LABEL_68;
      }
      else if ( v44 != (unsigned int)sub_C445E0((__int64)&v75) )
      {
LABEL_68:
        v45 = sub_AD8D80(v72, (__int64)&v75);
        v13 = v42;
        v79 = 1;
        v78 = 3;
        v77[0] = "mask";
        v42 = sub_B504D0(28, v42, v45, (__int64)v77, v30, 0);
        v48 = *(unsigned int *)(a4 + 8);
        if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v13 = a4 + 16;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v48 + 1, 8u, v46, v47);
          v48 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v48) = v42;
        ++*(_DWORD *)(a4 + 8);
      }
LABEL_71:
      if ( v11 != *(_QWORD *)(v42 + 8) )
      {
        v79 = 1;
        v77[0] = "zext";
        v78 = 3;
        v13 = a4;
        v49 = sub_B522D0(v42, v11, 0, (__int64)v77, v30, 0);
        v52 = *(unsigned int *)(a4 + 8);
        if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v13 = a4 + 16;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v52 + 1, 8u, v50, v51);
          v52 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v52) = v49;
        ++*(_DWORD *)(a4 + 8);
      }
      goto LABEL_42;
    }
  }
  return v9;
}
