// Function: sub_3024DF0
// Address: 0x3024df0
//
__int64 __fastcall sub_3024DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rax
  const char *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  char v14; // bl
  __int64 v15; // r12
  __int64 v16; // rsi
  bool v17; // bl
  __int64 v18; // r14
  void *v19; // r12
  size_t v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // r14
  _QWORD *v27; // rdx
  void *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  char v34; // al
  unsigned int v35; // esi
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int); // rax
  int v37; // eax
  int v38; // eax
  _WORD *v39; // rdx
  unsigned __int64 v40; // r14
  void *v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rdi
  _BYTE *v44; // rax
  __int64 v45; // r12
  unsigned __int16 v46; // ax
  unsigned __int8 v47; // bl
  unsigned __int16 v48; // ax
  unsigned int v49; // eax
  __int64 v50; // rcx
  void *v51; // rdx
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // r14
  _BYTE *v57; // rax
  char v58; // bl
  unsigned __int8 *v59; // rax
  size_t v60; // rdx
  unsigned __int64 v61; // r12
  _BYTE *v62; // rax
  unsigned __int16 v63; // ax
  unsigned __int8 v64; // bl
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r14
  unsigned __int8 *v70; // rax
  size_t v71; // rdx
  __int64 v72; // rax
  char v73; // cl
  unsigned __int8 *v74; // rax
  unsigned int v75; // eax
  unsigned int v76; // r14d
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned int v79; // eax
  __int64 v80; // r12
  unsigned __int16 v81; // ax
  unsigned __int64 v82; // rsi
  __int64 v83; // rdi
  __int64 v84; // rax
  __int64 v85; // rcx
  size_t v86; // rdx
  unsigned __int16 v87; // ax
  __int64 v89; // [rsp+10h] [rbp-C0h]
  __int64 v90; // [rsp+18h] [rbp-B8h]
  __int64 v91; // [rsp+20h] [rbp-B0h]
  __int64 v92; // [rsp+28h] [rbp-A8h]
  __int64 v93; // [rsp+38h] [rbp-98h]
  __int64 v94; // [rsp+40h] [rbp-90h]
  __int64 v95; // [rsp+48h] [rbp-88h]
  char v96; // [rsp+55h] [rbp-7Bh]
  char v97; // [rsp+56h] [rbp-7Ah]
  char v98; // [rsp+57h] [rbp-79h]
  void *s2; // [rsp+60h] [rbp-70h] BYREF
  size_t n; // [rsp+68h] [rbp-68h]
  _QWORD v102[2]; // [rsp+70h] [rbp-60h] BYREF
  unsigned __int8 *v103; // [rsp+80h] [rbp-50h] BYREF
  size_t v104; // [rsp+88h] [rbp-48h]
  _QWORD v105[8]; // [rsp+90h] [rbp-40h] BYREF

  v92 = sub_31DA930();
  v6 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 200) + 16LL))(*(_QWORD *)(a1 + 200), a3);
  v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 144LL);
  if ( v7 == sub_3020010 )
    v95 = v6 + 960;
  else
    v95 = v7(v6);
  v8 = *(_QWORD *)(a1 + 232);
  v89 = v8;
  if ( v8 )
    v89 = *(_QWORD *)(v8 + 40);
  v98 = sub_CE9220(a3);
  if ( !*(_QWORD *)(a3 + 104) )
  {
    v9 = "()";
    if ( !(*(_DWORD *)(*(_QWORD *)(a3 + 24) + 8LL) >> 8) )
      return sub_904010(a4, v9);
  }
  sub_904010(a4, "(\n");
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, (__int64)"(\n", v10, v11);
    v12 = *(_QWORD *)(a3 + 96);
    v94 = v12 + 40LL * *(_QWORD *)(a3 + 104);
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a3, (__int64)"(\n", v12 + 40LL * *(_QWORD *)(a3 + 104), v85);
      v12 = *(_QWORD *)(a3 + 96);
    }
  }
  else
  {
    v12 = *(_QWORD *)(a3 + 96);
    v94 = v12 + 40LL * *(_QWORD *)(a3 + 104);
  }
  if ( v94 == v12 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a3 + 24) + 8LL) >> 8 )
      goto LABEL_49;
    goto LABEL_51;
  }
  v13 = v12;
  v14 = 1;
  do
  {
    v15 = *(_QWORD *)(v13 + 8);
    v16 = v95;
    sub_3045B30(&s2, v95, a3, *(unsigned int *)(v13 + 32));
    if ( *(_BYTE *)(v15 + 8) != 15 )
    {
      if ( v14 )
        goto LABEL_13;
LABEL_69:
      v39 = *(_WORD **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v39 <= 1u )
      {
        v16 = (__int64)",\n";
        sub_CB6200(a4, (unsigned __int8 *)",\n", 2u);
      }
      else
      {
        v16 = 2604;
        *v39 = 2604;
        *(_QWORD *)(a4 + 32) += 2LL;
      }
      goto LABEL_13;
    }
    if ( (*(_BYTE *)(v15 + 9) & 1) == 0 )
      goto LABEL_43;
    if ( !v14 )
      goto LABEL_69;
LABEL_13:
    if ( v98 )
    {
      v96 = sub_CE8830((_BYTE *)v13);
      if ( v96 )
        goto LABEL_15;
      v97 = sub_CE8900((unsigned __int8 *)v13, v16);
      if ( v97 )
        goto LABEL_16;
      if ( (unsigned __int8)sub_CE8A00((unsigned __int8 *)v13, v16)
        || (unsigned __int8)sub_CE8980((unsigned __int8 *)v13, v16) )
      {
LABEL_15:
        v97 = 0;
LABEL_16:
        v17 = v98;
        if ( !v89 )
          goto LABEL_33;
        v18 = *(_QWORD *)(v89 + 8);
        v19 = s2;
        v20 = n;
        v21 = 32LL * *(unsigned int *)(v89 + 16);
        v91 = v18 + v21;
        v22 = v21 >> 5;
        v23 = v21 >> 7;
        if ( v23 )
        {
          v90 = a4;
          v24 = *(_QWORD *)(v89 + 8);
          v93 = v18 + (v23 << 7);
          do
          {
            if ( v20 == *(_QWORD *)(v24 + 8) && (!v20 || !memcmp(*(const void **)v24, v19, v20)) )
            {
              v26 = v24;
              a4 = v90;
              v17 = v26 == v91;
              goto LABEL_33;
            }
            if ( v20 == *(_QWORD *)(v24 + 40)
              && ((v25 = v24 + 32, !v20) || !memcmp(*(const void **)(v24 + 32), v19, v20))
              || v20 == *(_QWORD *)(v24 + 72)
              && ((v25 = v24 + 64, !v20) || !memcmp(*(const void **)(v24 + 64), v19, v20))
              || v20 == *(_QWORD *)(v24 + 104)
              && ((v25 = v24 + 96, !v20) || !memcmp(*(const void **)(v24 + 96), v19, v20)) )
            {
              a4 = v90;
              v17 = v91 == v25;
              goto LABEL_33;
            }
            v24 += 128;
          }
          while ( v93 != v24 );
          v18 = v24;
          a4 = v90;
          v22 = (v91 - v18) >> 5;
        }
        if ( v22 != 2 )
        {
          if ( v22 != 3 )
          {
            if ( v22 != 1 )
              goto LABEL_165;
LABEL_176:
            if ( v20 == *(_QWORD *)(v18 + 8) && (!v20 || !memcmp(*(const void **)v18, v19, v20)) )
              goto LABEL_179;
LABEL_165:
            v17 = v98;
            goto LABEL_33;
          }
          if ( v20 == *(_QWORD *)(v18 + 8) && (!v20 || !memcmp(*(const void **)v18, v19, v20)) )
          {
            v17 = v18 == v91;
            goto LABEL_33;
          }
          v18 += 32;
        }
        if ( v20 != *(_QWORD *)(v18 + 8) || v20 && memcmp(*(const void **)v18, v19, v20) )
        {
          v18 += 32;
          goto LABEL_176;
        }
LABEL_179:
        v17 = v91 == v18;
LABEL_33:
        if ( (unsigned __int8)sub_B2D790(v13, "nvvm.hidden", 0xBu) )
          sub_904010(a4, "\t.hidden");
        v27 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v27 <= 7u )
        {
          sub_CB6200(a4, "\t.param ", 8u);
        }
        else
        {
          *v27 = 0x206D617261702E09LL;
          *(_QWORD *)(a4 + 32) += 8LL;
        }
        if ( v17 )
          sub_904010(a4, ".u64 .ptr ");
        if ( v96 )
        {
          sub_904010(a4, ".samplerref ");
        }
        else if ( v97 )
        {
          sub_904010(a4, ".texref ");
        }
        else
        {
          sub_904010(a4, ".surfref ");
        }
        v14 = 0;
        sub_CB6200(a4, (unsigned __int8 *)s2, n);
LABEL_43:
        v28 = s2;
        if ( s2 == v102 )
          goto LABEL_45;
LABEL_44:
        j_j___libc_free_0((unsigned __int64)v28);
        goto LABEL_45;
      }
      if ( (unsigned __int8)sub_B2D680(v13) )
      {
        v45 = sub_B2BD20(v13);
        v46 = sub_CE9380(a3, *(_DWORD *)(v13 + 32) + 1);
        v47 = v46;
        if ( !HIBYTE(v46) )
        {
          v47 = sub_303E610(v95, a3, v45, v92);
          if ( (unsigned __int8)sub_B2D680(v13) )
          {
            v48 = sub_B2BD00(v13);
            if ( HIBYTE(v48) )
            {
              if ( v47 < (unsigned __int8)v48 )
                v47 = v48;
            }
          }
        }
LABEL_95:
        if ( (unsigned __int8)sub_B2D790(v13, "nvvm.hidden", 0xBu) )
          sub_904010(a4, "\t.hidden");
        v51 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v51 <= 0xEu )
        {
          v52 = sub_CB6200(a4, "\t.param .align ", 0xFu);
        }
        else
        {
          v52 = a4;
          qmemcpy(v51, "\t.param .align ", 15);
          *(_QWORD *)(a4 + 32) += 15LL;
        }
        v53 = sub_CB59D0(v52, 1LL << v47);
        v54 = *(_QWORD *)(v53 + 32);
        v55 = v53;
        if ( (unsigned __int64)(*(_QWORD *)(v53 + 24) - v54) <= 4 )
        {
          v55 = sub_CB6200(v53, " .b8 ", 5u);
        }
        else
        {
          *(_DWORD *)v54 = 945958432;
          *(_BYTE *)(v54 + 4) = 32;
          *(_QWORD *)(v53 + 32) += 5LL;
        }
        v56 = sub_CB6200(v55, (unsigned __int8 *)s2, n);
        v57 = *(_BYTE **)(v56 + 32);
        if ( *(_BYTE **)(v56 + 24) == v57 )
        {
          v56 = sub_CB6200(v56, (unsigned __int8 *)"[", 1u);
        }
        else
        {
          *v57 = 91;
          ++*(_QWORD *)(v56 + 32);
        }
        v58 = sub_AE5020(v92, v45);
        v59 = (unsigned __int8 *)sub_9208B0(v92, v45);
        v104 = v60;
        v103 = v59;
        v61 = ((1LL << v58) + ((unsigned __int64)(v59 + 7) >> 3) - 1) >> v58 << v58;
        if ( (_BYTE)v60 )
          sub_904010(v56, "vscale x ");
        sub_CB59D0(v56, v61);
        v62 = *(_BYTE **)(v56 + 32);
        if ( *(_BYTE **)(v56 + 24) == v62 )
        {
          v14 = 0;
          sub_CB6200(v56, (unsigned __int8 *)"]", 1u);
        }
        else
        {
          *v62 = 93;
          v14 = 0;
          ++*(_QWORD *)(v56 + 32);
        }
        goto LABEL_43;
      }
    }
    else if ( (unsigned __int8)sub_B2D680(v13) )
    {
      v45 = sub_B2BD20(v13);
      LOWORD(v49) = sub_B2BD00(v13);
      v50 = 0;
      if ( BYTE1(v49) )
        v50 = v49;
      v47 = sub_303F840(v95, a3, v45, v50, v92);
      goto LABEL_95;
    }
    v14 = sub_30201A0(v15);
    if ( v14 )
    {
      v63 = sub_CE9380(a3, *(_DWORD *)(v13 + 32) + 1);
      v64 = v63;
      if ( !HIBYTE(v63) )
      {
        v64 = sub_303E610(v95, a3, v15, v92);
        if ( (unsigned __int8)sub_B2D680(v13) )
        {
          v87 = sub_B2BD00(v13);
          if ( HIBYTE(v87) )
          {
            if ( v64 < (unsigned __int8)v87 )
              v64 = v87;
          }
        }
      }
      if ( (unsigned __int8)sub_B2D790(v13, "nvvm.hidden", 0xBu) )
        sub_904010(a4, "\t.hidden");
      v65 = sub_904010(a4, "\t.param .align ");
      v66 = sub_CB59D0(v65, 1LL << v64);
      v67 = sub_904010(v66, " .b8 ");
      v68 = sub_CB6200(v67, (unsigned __int8 *)s2, n);
      v69 = sub_904010(v68, "[");
      v70 = (unsigned __int8 *)sub_BDB740(v92, v15);
      v104 = v71;
      v103 = v70;
      if ( (_BYTE)v71 )
        sub_904010(v69, "vscale x ");
      v14 = 0;
      sub_CB59D0(v69, (unsigned __int64)v103);
      sub_904010(v69, "]");
      goto LABEL_43;
    }
    v34 = *(_BYTE *)(v15 + 8);
    if ( v34 == 14 )
    {
      v35 = *(_DWORD *)(v15 + 8) >> 8;
      v36 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v95 + 32LL);
      if ( v36 == sub_2D42F30 )
      {
        v37 = sub_AE2980(v92, v35)[1];
        switch ( v37 )
        {
          case 1:
            v38 = 2;
            break;
          case 2:
            v38 = 3;
            break;
          case 4:
            v38 = 4;
            break;
          case 8:
            v38 = 5;
            break;
          case 16:
            v38 = 6;
            break;
          case 32:
            v38 = 7;
            break;
          case 64:
            v38 = 8;
            break;
          case 128:
            v38 = 9;
            break;
          default:
            goto LABEL_188;
        }
      }
      else
      {
        v38 = (unsigned __int16)v36(v95, v92, v35);
        if ( (unsigned __int16)v38 <= 1u || (unsigned __int16)(v38 - 504) <= 7u )
LABEL_188:
          BUG();
      }
      v72 = 16LL * (v38 - 1);
      v73 = byte_444C4A0[v72 + 8];
      v74 = *(unsigned __int8 **)&byte_444C4A0[v72];
      LOBYTE(v104) = v73;
      v103 = v74;
      v75 = sub_CA1930(&v103);
      v76 = v75;
      if ( !v98 )
      {
        v40 = v75;
        if ( *(_BYTE *)(v15 + 8) != 12 )
          goto LABEL_76;
        goto LABEL_73;
      }
      if ( (unsigned __int8)sub_B2D790(v13, "nvvm.hidden", 0xBu) )
        sub_904010(a4, "\t.hidden");
      v77 = sub_904010(a4, "\t.param .u");
      v78 = sub_CB59D0(v77, v76);
      sub_904010(v78, " .ptr");
      v79 = *(_DWORD *)(v15 + 8) >> 8;
      if ( v79 == 4 )
      {
        sub_904010(a4, " .const");
      }
      else if ( v79 > 4 )
      {
        if ( v79 == 5 )
          sub_904010(a4, " .local");
      }
      else if ( v79 == 1 )
      {
        sub_904010(a4, " .global");
      }
      else if ( v79 == 3 )
      {
        sub_904010(a4, " .shared");
      }
      v80 = sub_904010(a4, " .align ");
      v81 = sub_B2BD00(v13);
      v82 = 1;
      if ( HIBYTE(v81) )
        v82 = 1LL << v81;
      v83 = sub_CB59D0(v80, v82);
LABEL_137:
      v84 = sub_904010(v83, " ");
      sub_CB6200(v84, (unsigned __int8 *)s2, n);
      goto LABEL_43;
    }
    if ( v98 )
    {
      if ( (unsigned __int8)sub_B2D790(v13, "nvvm.hidden", 0xBu) )
        sub_904010(a4, "\t.hidden");
      sub_904010(a4, "\t.param .");
      if ( sub_BCAC40(v15, 1) )
      {
        sub_904010(a4, "u8");
      }
      else
      {
        sub_30246F0((__int64)&v103, a1, v15, 1);
        sub_CB6200(a4, v103, v104);
        if ( v103 != (unsigned __int8 *)v105 )
          j_j___libc_free_0((unsigned __int64)v103);
      }
      v83 = a4;
      goto LABEL_137;
    }
    if ( v34 != 12 )
    {
      v103 = (unsigned __int8 *)sub_BCAE30(v15);
      v104 = v86;
      v40 = (unsigned int)sub_CA1930(&v103);
      goto LABEL_76;
    }
LABEL_73:
    v40 = 32;
    if ( *(_DWORD *)(v15 + 8) > 0x20FFu )
    {
      v40 = 64;
      if ( *(_DWORD *)(v15 + 8) >> 8 >= 0x40u )
        v40 = *(_DWORD *)(v15 + 8) >> 8;
    }
LABEL_76:
    v41 = *(void **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v41 <= 9u )
    {
      v42 = sub_CB6200(a4, "\t.param .b", 0xAu);
    }
    else
    {
      v42 = a4;
      qmemcpy(v41, "\t.param .b", 10);
      *(_QWORD *)(a4 + 32) += 10LL;
    }
    v43 = sub_CB59D0(v42, v40);
    v44 = *(_BYTE **)(v43 + 32);
    if ( *(_BYTE **)(v43 + 24) == v44 )
    {
      v43 = sub_CB6200(v43, (unsigned __int8 *)" ", 1u);
    }
    else
    {
      *v44 = 32;
      ++*(_QWORD *)(v43 + 32);
    }
    sub_CB6200(v43, (unsigned __int8 *)s2, n);
    v28 = s2;
    if ( s2 != v102 )
      goto LABEL_44;
LABEL_45:
    v13 += 40;
  }
  while ( v94 != v13 );
  if ( *(_DWORD *)(*(_QWORD *)(a3 + 24) + 8LL) >> 8 )
  {
    if ( !v14 )
      sub_904010(a4, ",\n");
LABEL_49:
    v29 = sub_904010(a4, "\t.param .align ");
    v30 = sub_CB59D0(v29, 8u);
    v31 = sub_904010(v30, " .b8 ");
    sub_3045B30(&v103, v95, a3, 0xFFFFFFFFLL);
    v32 = sub_CB6200(v31, v103, v104);
    sub_904010(v32, "[]");
    if ( v103 != (unsigned __int8 *)v105 )
      j_j___libc_free_0((unsigned __int64)v103);
  }
LABEL_51:
  v9 = "\n)";
  return sub_904010(a4, v9);
}
