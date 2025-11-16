// Function: sub_17A8400
// Address: 0x17a8400
//
__int64 *__fastcall sub_17A8400(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 *a6, __int64 a7)
{
  unsigned int v10; // r15d
  bool v11; // al
  __int64 **v13; // rax
  __int64 *v14; // r12
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned int v17; // r9d
  _QWORD *v18; // r15
  unsigned int v19; // r14d
  _QWORD *v20; // rbx
  unsigned int v21; // eax
  unsigned int v22; // edx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // eax
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // rax
  __int128 v30; // rax
  unsigned __int64 v31; // rax
  unsigned int v32; // r14d
  int v33; // r8d
  unsigned __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  unsigned __int64 v40; // r8
  unsigned __int64 v41; // r8
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  signed __int64 v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rsi
  unsigned __int8 *v48; // rsi
  __int64 v49; // rcx
  int v50; // eax
  char v51; // al
  unsigned int v52; // eax
  unsigned int v53; // esi
  unsigned __int64 v54; // rax
  unsigned int v55; // eax
  unsigned int v56; // ecx
  unsigned __int64 v57; // rax
  int v58; // r8d
  __int64 v59; // rax
  bool v60; // al
  bool v61; // al
  unsigned int v62; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v63; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v64; // [rsp+8h] [rbp-C8h]
  char v65; // [rsp+8h] [rbp-C8h]
  unsigned int v66; // [rsp+8h] [rbp-C8h]
  unsigned int v67; // [rsp+8h] [rbp-C8h]
  unsigned int v70; // [rsp+20h] [rbp-B0h]
  const void *v71; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v72; // [rsp+20h] [rbp-B0h]
  unsigned int v73; // [rsp+20h] [rbp-B0h]
  unsigned int v74; // [rsp+20h] [rbp-B0h]
  unsigned int v75; // [rsp+20h] [rbp-B0h]
  bool v77; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v78; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v79; // [rsp+38h] [rbp-98h]
  unsigned __int64 v80; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v81; // [rsp+48h] [rbp-88h]
  const void *v82; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v83; // [rsp+58h] [rbp-78h]
  const void *v84; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v85; // [rsp+68h] [rbp-68h]
  unsigned __int64 v86; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v87; // [rsp+78h] [rbp-58h]
  unsigned __int64 v88; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v89; // [rsp+88h] [rbp-48h]
  __int16 v90; // [rsp+90h] [rbp-40h]

  v10 = *(_DWORD *)(a5 + 8);
  if ( v10 <= 0x40 )
    v11 = *(_QWORD *)a5 == 0;
  else
    v11 = v10 == (unsigned int)sub_16A57B0(a5);
  if ( v11 )
    return 0;
  v70 = *(_DWORD *)(a3 + 8);
  if ( v70 <= 0x40 ? *(_QWORD *)a3 == 0 : v70 == (unsigned int)sub_16A57B0(a3) )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v13 = *(__int64 ***)(a2 - 8);
  else
    v13 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v14 = *v13;
  v15 = sub_16431D0(**v13);
  v16 = v15;
  v17 = v15;
  if ( v10 > 0x40 )
  {
    v62 = v15;
    v64 = v15;
    v26 = sub_16A57B0(a5);
    v16 = v64;
    v17 = v62;
    if ( v10 - v26 > 0x40 )
      return 0;
    v18 = **(_QWORD ***)a5;
  }
  else
  {
    v18 = *(_QWORD **)a5;
  }
  if ( v16 <= (unsigned __int64)v18 )
    return 0;
  v19 = v70;
  if ( v70 > 0x40 )
  {
    v66 = v17;
    v72 = v16;
    v50 = sub_16A57B0(a3);
    v16 = v72;
    v17 = v66;
    if ( v19 - v50 > 0x40 )
      return 0;
    v20 = **(_QWORD ***)a3;
  }
  else
  {
    v20 = *(_QWORD **)a3;
  }
  if ( v16 <= (unsigned __int64)v20 )
    return 0;
  v21 = *(_DWORD *)(a7 + 24);
  if ( v21 > 0x40 )
  {
    v74 = v17;
    memset(*(void **)(a7 + 16), 0, 8 * (((unsigned __int64)v21 + 63) >> 6));
    v17 = v74;
  }
  else
  {
    *(_QWORD *)(a7 + 16) = 0;
  }
  v22 = (_DWORD)v18 - 1;
  if ( (_DWORD)v18 != 1 )
  {
    if ( v22 > 0x40 )
    {
      v75 = v17;
      sub_16A5260((_QWORD *)a7, 0, v22);
      v28 = (unsigned __int64 *)a7;
      v17 = v75;
      if ( *(_DWORD *)(a7 + 8) <= 0x40u )
        goto LABEL_27;
LABEL_94:
      v73 = v17;
      sub_16A8890((__int64 *)a7, a6);
      v17 = v73;
      v79 = v73;
      if ( v73 <= 0x40 )
        goto LABEL_29;
      goto LABEL_95;
    }
    v23 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v18);
    v24 = *(_QWORD *)a7;
    if ( *(_DWORD *)(a7 + 8) <= 0x40u )
    {
      v25 = v23 | v24;
      *(_QWORD *)a7 = v25;
      goto LABEL_28;
    }
    *(_QWORD *)v24 |= v23;
  }
  v28 = (unsigned __int64 *)a7;
  if ( *(_DWORD *)(a7 + 8) > 0x40u )
    goto LABEL_94;
LABEL_27:
  v25 = *v28;
LABEL_28:
  v79 = v17;
  *(_QWORD *)a7 = *a6 & v25;
  if ( v17 <= 0x40 )
  {
LABEL_29:
    v81 = v17;
    v78 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    v80 = v78;
    v65 = *(_BYTE *)(a2 + 16);
    if ( v65 != 48 )
    {
      v89 = v17;
      goto LABEL_31;
    }
    v89 = v17;
LABEL_121:
    v88 = v78;
    v55 = v89;
    goto LABEL_122;
  }
LABEL_95:
  v67 = v17;
  sub_16A4EF0((__int64)&v78, -1, 1);
  v81 = v67;
  sub_16A4EF0((__int64)&v80, -1, 1);
  v51 = *(_BYTE *)(a2 + 16);
  v17 = v79;
  v65 = v51;
  v89 = v79;
  if ( v51 != 48 )
  {
    if ( v79 > 0x40 )
    {
      sub_16A4FD0((__int64)&v88, (const void **)&v78);
      v17 = v89;
      if ( v89 > 0x40 )
      {
        sub_16A5E70((__int64)&v88, (unsigned int)v20);
        v17 = v89;
        v87 = v89;
        if ( v89 > 0x40 )
        {
          sub_16A4FD0((__int64)&v86, (const void **)&v88);
          v17 = v87;
          if ( v87 > 0x40 )
          {
            sub_16A7DC0((__int64 *)&v86, (unsigned int)v18);
            goto LABEL_39;
          }
LABEL_36:
          v31 = 0;
          if ( (_DWORD)v18 != v17 )
            v31 = (v86 << (char)v18) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
          v86 = v31;
LABEL_39:
          if ( v79 > 0x40 && v78 )
            j_j___libc_free_0_0(v78);
          v78 = v86;
          v79 = v87;
          if ( v89 > 0x40 && v88 )
            j_j___libc_free_0_0(v88);
          v32 = v81;
          if ( (unsigned int)v18 < (unsigned int)v20 )
          {
            v89 = v81;
            v33 = (_DWORD)v20 - (_DWORD)v18;
            if ( v81 <= 0x40 )
            {
              v34 = v80;
              v35 = v81;
LABEL_48:
              v36 = (__int64)(v34 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
              v37 = v36 >> v33;
              v38 = v36 >> 63;
              if ( v33 == v32 )
                v37 = v38;
              v88 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & v37;
              goto LABEL_51;
            }
            sub_16A4FD0((__int64)&v88, (const void **)&v80);
            v32 = v89;
            v33 = (_DWORD)v20 - (_DWORD)v18;
            if ( v89 <= 0x40 )
            {
              v34 = v88;
              v35 = v81;
              goto LABEL_48;
            }
            sub_16A5E70((__int64)&v88, (_DWORD)v20 - (_DWORD)v18);
            v35 = v81;
LABEL_51:
            if ( v35 > 0x40 && v80 )
              j_j___libc_free_0_0(v80);
            v32 = v89;
            v80 = v88;
            v81 = v89;
            goto LABEL_55;
          }
          goto LABEL_112;
        }
LABEL_35:
        v86 = v88;
        goto LABEL_36;
      }
      v29 = v88;
LABEL_32:
      v87 = v17;
      v30 = (__int64)(v29 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
      *(_QWORD *)&v30 = (__int64)v30 >> (char)v20;
      if ( (_DWORD)v20 == v17 )
        *(_QWORD *)&v30 = *((_QWORD *)&v30 + 1);
      v88 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & v30;
      goto LABEL_35;
    }
LABEL_31:
    v29 = v78;
    goto LABEL_32;
  }
  if ( v79 <= 0x40 )
    goto LABEL_121;
  sub_16A4FD0((__int64)&v88, (const void **)&v78);
  v17 = v89;
  v55 = v89;
  if ( v89 > 0x40 )
  {
    sub_16A8110((__int64)&v88, (unsigned int)v20);
    v55 = v89;
    goto LABEL_124;
  }
LABEL_122:
  if ( (_DWORD)v20 == v17 )
    v88 = 0;
  else
    v88 >>= (char)v20;
LABEL_124:
  v87 = v55;
  v56 = v55;
  if ( v55 <= 0x40 )
  {
    v86 = v88;
LABEL_126:
    v57 = 0;
    if ( (_DWORD)v18 != v56 )
      v57 = (v86 << (char)v18) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v56);
    v86 = v57;
    goto LABEL_129;
  }
  sub_16A4FD0((__int64)&v86, (const void **)&v88);
  v56 = v87;
  if ( v87 <= 0x40 )
    goto LABEL_126;
  sub_16A7DC0((__int64 *)&v86, (unsigned int)v18);
LABEL_129:
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  v78 = v86;
  v79 = v87;
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  v32 = v81;
  if ( (unsigned int)v18 < (unsigned int)v20 )
  {
    v89 = v81;
    v58 = (_DWORD)v20 - (_DWORD)v18;
    if ( v81 > 0x40 )
    {
      sub_16A4FD0((__int64)&v88, (const void **)&v80);
      v32 = v89;
      v58 = (_DWORD)v20 - (_DWORD)v18;
      if ( v89 > 0x40 )
      {
        sub_16A8110((__int64)&v88, (_DWORD)v20 - (_DWORD)v18);
        v65 = 48;
        v35 = v81;
        goto LABEL_51;
      }
    }
    else
    {
      v88 = v80;
    }
    if ( v58 == v32 )
    {
      v88 = 0;
      v35 = v81;
      v65 = 48;
    }
    else
    {
      v65 = 48;
      v35 = v81;
      v88 >>= v58;
    }
    goto LABEL_51;
  }
  v65 = 48;
LABEL_112:
  v53 = (_DWORD)v18 - (_DWORD)v20;
  if ( v32 <= 0x40 )
  {
    v54 = 0;
    if ( v53 != v32 )
      v54 = (v80 << v53) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v32);
    v80 = v54;
    goto LABEL_56;
  }
  sub_16A7DC0((__int64 *)&v80, v53);
  v32 = v81;
LABEL_55:
  v85 = v32;
  if ( v32 <= 0x40 )
  {
LABEL_56:
    v39 = v80;
LABEL_57:
    v71 = (const void *)(*a6 & v39);
    v84 = v71;
    goto LABEL_58;
  }
  sub_16A4FD0((__int64)&v84, (const void **)&v80);
  v32 = v85;
  if ( v85 <= 0x40 )
  {
    v39 = (__int64)v84;
    goto LABEL_57;
  }
  sub_16A8890((__int64 *)&v84, a6);
  v32 = v85;
  v71 = v84;
LABEL_58:
  v83 = v32;
  v85 = 0;
  v82 = v71;
  v89 = v79;
  if ( v79 <= 0x40 )
  {
    v40 = v78;
LABEL_60:
    v41 = *a6 & v40;
    goto LABEL_61;
  }
  sub_16A4FD0((__int64)&v88, (const void **)&v78);
  if ( v89 <= 0x40 )
  {
    v40 = v88;
    goto LABEL_60;
  }
  sub_16A8890((__int64 *)&v88, a6);
  v52 = v89;
  v41 = v88;
  v89 = 0;
  v87 = v52;
  v86 = v88;
  if ( v52 > 0x40 )
  {
    v63 = v88;
    v77 = sub_16A5220((__int64)&v86, &v82);
    if ( v63 )
    {
      j_j___libc_free_0_0(v63);
      if ( v89 > 0x40 )
      {
        if ( v88 )
          j_j___libc_free_0_0(v88);
      }
    }
    goto LABEL_62;
  }
LABEL_61:
  v77 = v71 == (const void *)v41;
LABEL_62:
  if ( v32 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  if ( v77 )
  {
    if ( (_DWORD)v18 == (_DWORD)v20 )
      goto LABEL_83;
    v42 = *(_QWORD *)(a2 + 8);
    if ( v42 )
    {
      if ( !*(_QWORD *)(v42 + 8) )
      {
        v43 = *v14;
        if ( (unsigned int)v18 > (unsigned int)v20 )
        {
          v59 = sub_15A0680(v43, (unsigned int)((_DWORD)v18 - (_DWORD)v20), 0);
          v90 = 257;
          v14 = (__int64 *)sub_15FB440(23, v14, v59, (__int64)&v88, 0);
          v60 = sub_15F2380((__int64)a4);
          sub_15F2330((__int64)v14, v60);
          v61 = sub_15F2370((__int64)a4);
          sub_15F2310((__int64)v14, v61);
        }
        else
        {
          v44 = sub_15A0680(v43, (unsigned int)((_DWORD)v20 - (_DWORD)v18), 0);
          v90 = 257;
          if ( v65 == 48 )
            v14 = (__int64 *)sub_15FB440(24, v14, v44, (__int64)&v88, 0);
          else
            v14 = (__int64 *)sub_15FB440(25, v14, v44, (__int64)&v88, 0);
          if ( sub_15F23D0(a2) )
            sub_15F2350((__int64)v14, 1);
        }
        v45 = a4[6];
        v88 = v45;
        if ( v45 )
        {
          v46 = (__int64)(v14 + 6);
          sub_1623A60((__int64)&v88, v45, 2);
          v47 = v14[6];
          if ( !v47 )
            goto LABEL_80;
        }
        else
        {
          v47 = v14[6];
          v46 = (__int64)(v14 + 6);
          if ( !v47 )
          {
LABEL_82:
            sub_157E9D0(a4[5] + 40LL, (__int64)v14);
            v49 = a4[3];
            v14[4] = (__int64)(a4 + 3);
            v49 &= 0xFFFFFFFFFFFFFFF8LL;
            v14[3] = v49 | v14[3] & 7;
            *(_QWORD *)(v49 + 8) = v14 + 3;
            a4[3] = a4[3] & 7LL | (unsigned __int64)(v14 + 3);
            sub_170B990(*a1, (__int64)v14);
            goto LABEL_83;
          }
        }
        sub_161E7C0(v46, v47);
LABEL_80:
        v48 = (unsigned __int8 *)v88;
        v14[6] = v88;
        if ( v48 )
          sub_1623210((__int64)&v88, v48, v46);
        goto LABEL_82;
      }
    }
  }
  v14 = 0;
LABEL_83:
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  return v14;
}
