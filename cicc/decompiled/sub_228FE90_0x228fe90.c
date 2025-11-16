// Function: sub_228FE90
// Address: 0x228fe90
//
__int64 __fastcall sub_228FE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // ebx
  __int64 v10; // r12
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  _QWORD *v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rax
  __int16 v20; // dx
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  _QWORD *v25; // rcx
  _QWORD *v26; // r14
  __int16 v27; // ax
  unsigned int v28; // eax
  unsigned int v29; // r15d
  __int64 *v30; // r13
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v37; // eax
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // rbx
  __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 *v45; // rsi
  __int64 v46; // rax
  __int16 v47; // dx
  __int64 v48; // rsi
  __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int16 v53; // dx
  __int64 v54; // rsi
  _QWORD *v55; // rax
  __int16 v56; // dx
  _BYTE *v57; // rax
  __int64 v58; // [rsp+0h] [rbp-120h]
  __int64 v59; // [rsp+8h] [rbp-118h]
  _QWORD *v60; // [rsp+18h] [rbp-108h]
  unsigned int v61; // [rsp+3Ch] [rbp-E4h]
  _QWORD **v63; // [rsp+50h] [rbp-D0h]
  _QWORD *v66; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v67; // [rsp+78h] [rbp-A8h]
  _QWORD *v68; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v69; // [rsp+88h] [rbp-98h]
  __int64 v70[2]; // [rsp+90h] [rbp-90h] BYREF
  __int64 v71[2]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v72; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v73; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v74; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v75; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v76; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v77; // [rsp+D8h] [rbp-48h]
  _QWORD *v78; // [rsp+E0h] [rbp-40h] BYREF
  unsigned int v79; // [rsp+E8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_D95540(a2);
  v67 = sub_D97050(v4, v5);
  v9 = v67;
  if ( v67 > 0x40 )
    sub_C43690((__int64)&v66, 0, 0);
  else
    v66 = 0;
  v10 = a2;
  if ( *(_WORD *)(a2 + 24) != 8 )
  {
    v18 = a3;
    v10 = a2;
    if ( *(_WORD *)(a3 + 24) == 8 )
      goto LABEL_29;
    goto LABEL_71;
  }
  do
  {
    v11 = sub_D33D80((_QWORD *)v10, *(_QWORD *)(a1 + 8), v6, v7, v8);
    v12 = *(_WORD *)(v11 + 24);
    if ( v12 )
    {
      if ( v12 != 6 || (v11 = **(_QWORD **)(v11 + 32), *(_WORD *)(v11 + 24)) )
      {
LABEL_76:
        v29 = 0;
        goto LABEL_111;
      }
    }
    v13 = *(_QWORD *)(v11 + 32);
    v14 = *(_DWORD *)(v13 + 32);
    v73 = v14;
    if ( v14 <= 0x40 )
    {
      v72 = *(_QWORD *)(v13 + 24);
      v15 = 1LL << ((unsigned __int8)v14 - 1);
LABEL_7:
      v16 = v72;
      if ( (v15 & v72) != 0 )
      {
        v79 = v14;
        goto LABEL_9;
      }
      v75 = v14;
      v74 = v72;
LABEL_13:
      v77 = v67;
      if ( v67 > 0x40 )
        goto LABEL_62;
      goto LABEL_14;
    }
    sub_C43780((__int64)&v72, (const void **)(v13 + 24));
    v14 = v73;
    v15 = 1LL << ((unsigned __int8)v73 - 1);
    if ( v73 <= 0x40 )
      goto LABEL_7;
    if ( (*(_QWORD *)(v72 + 8LL * ((v73 - 1) >> 6)) & v15) != 0 )
    {
      v79 = v73;
      sub_C43780((__int64)&v78, (const void **)&v72);
      v14 = v79;
      if ( v79 > 0x40 )
      {
        sub_C43D10((__int64)&v78);
        goto LABEL_12;
      }
      v16 = (unsigned __int64)v78;
LABEL_9:
      v17 = (_QWORD *)(~v16 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14));
      if ( !v14 )
        v17 = 0;
      v78 = v17;
LABEL_12:
      sub_C46250((__int64)&v78);
      v75 = v79;
      v74 = (unsigned __int64)v78;
      goto LABEL_13;
    }
    v75 = v73;
    sub_C43780((__int64)&v74, (const void **)&v72);
    v77 = v67;
    if ( v67 > 0x40 )
    {
LABEL_62:
      sub_C43780((__int64)&v76, (const void **)&v66);
      goto LABEL_15;
    }
LABEL_14:
    v76 = (unsigned __int64)v66;
LABEL_15:
    sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
    if ( v67 > 0x40 && v66 )
      j_j___libc_free_0_0((unsigned __int64)v66);
    v66 = v78;
    v67 = v79;
    if ( v77 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
    if ( v75 > 0x40 && v74 )
      j_j___libc_free_0_0(v74);
    v10 = **(_QWORD **)(v10 + 32);
    if ( v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
  }
  while ( *(_WORD *)(v10 + 24) == 8 );
  v18 = a3;
  if ( *(_WORD *)(a3 + 24) == 8 )
  {
    while ( 1 )
    {
LABEL_29:
      v19 = sub_D33D80((_QWORD *)v18, *(_QWORD *)(a1 + 8), v6, v7, v8);
      v20 = *(_WORD *)(v19 + 24);
      if ( v20 )
      {
        if ( v20 != 6 )
          goto LABEL_76;
        v19 = **(_QWORD **)(v19 + 32);
        if ( *(_WORD *)(v19 + 24) )
          goto LABEL_76;
      }
      v21 = *(_QWORD *)(v19 + 32);
      v22 = *(_DWORD *)(v21 + 32);
      v73 = v22;
      if ( v22 <= 0x40 )
        break;
      sub_C43780((__int64)&v72, (const void **)(v21 + 24));
      v22 = v73;
      v23 = 1LL << ((unsigned __int8)v73 - 1);
      if ( v73 <= 0x40 )
        goto LABEL_32;
      if ( (*(_QWORD *)(v72 + 8LL * ((v73 - 1) >> 6)) & v23) != 0 )
      {
        v79 = v73;
        sub_C43780((__int64)&v78, (const void **)&v72);
        v22 = v79;
        if ( v79 > 0x40 )
        {
          sub_C43D10((__int64)&v78);
          goto LABEL_37;
        }
        v24 = (unsigned __int64)v78;
LABEL_34:
        v25 = (_QWORD *)(~v24 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22));
        if ( !v22 )
          v25 = 0;
        v78 = v25;
LABEL_37:
        sub_C46250((__int64)&v78);
        v75 = v79;
        v74 = (unsigned __int64)v78;
        goto LABEL_38;
      }
      v75 = v73;
      sub_C43780((__int64)&v74, (const void **)&v72);
      v77 = v67;
      if ( v67 <= 0x40 )
      {
LABEL_39:
        v76 = (unsigned __int64)v66;
        goto LABEL_40;
      }
LABEL_69:
      sub_C43780((__int64)&v76, (const void **)&v66);
LABEL_40:
      sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
      if ( v67 > 0x40 && v66 )
        j_j___libc_free_0_0((unsigned __int64)v66);
      v66 = v78;
      v67 = v79;
      if ( v77 > 0x40 && v76 )
        j_j___libc_free_0_0(v76);
      if ( v75 > 0x40 && v74 )
        j_j___libc_free_0_0(v74);
      v18 = **(_QWORD **)(v18 + 32);
      if ( v73 > 0x40 && v72 )
        j_j___libc_free_0_0(v72);
      if ( *(_WORD *)(v18 + 24) != 8 )
      {
        v69 = v9;
        if ( v9 <= 0x40 )
          goto LABEL_54;
LABEL_72:
        sub_C43690((__int64)&v68, 0, 0);
        goto LABEL_55;
      }
    }
    v72 = *(_QWORD *)(v21 + 24);
    v23 = 1LL << ((unsigned __int8)v22 - 1);
LABEL_32:
    v24 = v72;
    if ( (v23 & v72) != 0 )
    {
      v79 = v22;
      goto LABEL_34;
    }
    v75 = v22;
    v74 = v72;
LABEL_38:
    v77 = v67;
    if ( v67 <= 0x40 )
      goto LABEL_39;
    goto LABEL_69;
  }
LABEL_71:
  v69 = v9;
  v18 = a3;
  if ( v9 > 0x40 )
    goto LABEL_72;
LABEL_54:
  v68 = 0;
LABEL_55:
  v26 = sub_DCC810(*(__int64 **)(a1 + 8), v18, v10, 0, 0);
  v27 = *((_WORD *)v26 + 12);
  if ( v27 )
  {
    if ( v27 != 5 )
      goto LABEL_107;
    v30 = (__int64 *)v26[4];
    v31 = &v30[v26[5]];
    if ( v30 == v31 )
      goto LABEL_107;
    v26 = 0;
    do
    {
      while ( 1 )
      {
        v32 = *v30;
        v33 = *(_WORD *)(*v30 + 24);
        if ( v33 )
          break;
        ++v30;
        v26 = (_QWORD *)v32;
        if ( v31 == v30 )
          goto LABEL_106;
      }
      if ( v33 != 6 )
        goto LABEL_107;
      v34 = **(_QWORD **)(v32 + 32);
      if ( *(_WORD *)(v34 + 24) )
        goto LABEL_107;
      v35 = *(_QWORD *)(v34 + 32);
      v73 = *(_DWORD *)(v35 + 32);
      if ( v73 > 0x40 )
        sub_C43780((__int64)&v72, (const void **)(v35 + 24));
      else
        v72 = *(_QWORD *)(v35 + 24);
      sub_9692E0((__int64)&v74, (__int64 *)&v72);
      v77 = v69;
      if ( v69 > 0x40 )
        sub_C43780((__int64)&v76, (const void **)&v68);
      else
        v76 = (unsigned __int64)v68;
      sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
      if ( v69 > 0x40 && v68 )
        j_j___libc_free_0_0((unsigned __int64)v68);
      v68 = v78;
      v69 = v79;
      if ( v77 > 0x40 && v76 )
        j_j___libc_free_0_0(v76);
      if ( v75 > 0x40 && v74 )
        j_j___libc_free_0_0(v74);
      if ( v73 > 0x40 && v72 )
        j_j___libc_free_0_0(v72);
      ++v30;
    }
    while ( v31 != v30 );
LABEL_106:
    if ( !v26 )
    {
LABEL_107:
      v29 = 0;
      goto LABEL_108;
    }
  }
  sub_9865C0((__int64)v70, v26[4] + 24LL);
  LOBYTE(v28) = sub_D94970((__int64)v70, 0);
  v29 = v28;
  if ( (_BYTE)v28 )
  {
    v29 = 0;
    goto LABEL_58;
  }
  sub_9865C0((__int64)&v76, (__int64)&v68);
  sub_9865C0((__int64)&v74, (__int64)&v66);
  sub_C49E90((__int64)&v78, (__int64)&v74, (__int64)&v76);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0((unsigned __int64)v66);
  v66 = v78;
  v37 = v79;
  v79 = 0;
  v67 = v37;
  sub_969240((__int64 *)&v78);
  sub_969240((__int64 *)&v74);
  sub_969240((__int64 *)&v76);
  sub_C4B8A0((__int64)v71, (__int64)v70, (__int64)&v66);
  if ( !sub_D94970((__int64)v71, 0) )
  {
    v29 = 1;
    goto LABEL_187;
  }
  if ( *(_WORD *)(a2 + 24) != 8 )
    goto LABEL_187;
  v40 = a2;
  while ( 2 )
  {
    v59 = **(_QWORD **)(v40 + 32);
    v63 = *(_QWORD ***)(v40 + 48);
    if ( v67 <= 0x40 && v69 <= 0x40 )
    {
      v41 = (__int64)v68;
      v67 = v69;
      v66 = v68;
    }
    else
    {
      sub_C43990((__int64)&v66, (__int64)&v68);
    }
    v58 = sub_D33D80((_QWORD *)v40, *(_QWORD *)(a1 + 8), v41, v38, v39);
    v42 = a2;
    v60 = sub_DCC810(*(__int64 **)(a1 + 8), v58, v58, 0, 0);
    while ( 1 )
    {
      v49 = v67;
      if ( v67 <= 0x40 )
      {
        if ( v66 == (_QWORD *)1 )
          goto LABEL_151;
      }
      else
      {
        v61 = v67;
        v50 = sub_C444A0((__int64)&v66);
        v49 = v61;
        v44 = v50;
        if ( v61 - v50 <= 0x40 && *v66 == 1 )
        {
LABEL_151:
          v51 = a3;
          v45 = *(__int64 **)(a1 + 8);
          goto LABEL_173;
        }
      }
      v45 = *(__int64 **)(a1 + 8);
      if ( *(_WORD *)(v42 + 24) != 8 )
        break;
      v46 = sub_D33D80((_QWORD *)v42, (__int64)v45, v49, v43, v44);
      v43 = (__int64)v63;
      if ( v63 != *(_QWORD ***)(v42 + 48) )
      {
        v47 = *(_WORD *)(v46 + 24);
        if ( v47 )
        {
          if ( v47 != 6 )
            goto LABEL_186;
          v46 = **(_QWORD **)(v46 + 32);
          if ( *(_WORD *)(v46 + 24) )
            goto LABEL_186;
        }
        v48 = *(_QWORD *)(v46 + 32);
        v73 = *(_DWORD *)(v48 + 32);
        if ( v73 > 0x40 )
          sub_C43780((__int64)&v72, (const void **)(v48 + 24));
        else
          v72 = *(_QWORD *)(v48 + 24);
        sub_9692E0((__int64)&v74, (__int64 *)&v72);
        v77 = v67;
        if ( v67 > 0x40 )
          sub_C43780((__int64)&v76, (const void **)&v66);
        else
          v76 = (unsigned __int64)v66;
        sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
        if ( v67 > 0x40 && v66 )
          j_j___libc_free_0_0((unsigned __int64)v66);
        v66 = v78;
        v67 = v79;
        if ( v77 > 0x40 && v76 )
          j_j___libc_free_0_0(v76);
        if ( v75 > 0x40 && v74 )
          j_j___libc_free_0_0(v74);
        if ( v73 > 0x40 && v72 )
          j_j___libc_free_0_0(v72);
      }
      v42 = **(_QWORD **)(v42 + 32);
    }
    v51 = a3;
LABEL_173:
    while ( 2 )
    {
      if ( (unsigned int)v49 <= 0x40 )
      {
        if ( v66 == (_QWORD *)1 )
          break;
        goto LABEL_153;
      }
      v49 = (unsigned int)v49 - (unsigned int)sub_C444A0((__int64)&v66);
      if ( (unsigned int)v49 > 0x40 || *v66 != 1 )
      {
LABEL_153:
        if ( *(_WORD *)(v51 + 24) != 8 )
          break;
        v52 = sub_D33D80((_QWORD *)v51, (__int64)v45, v49, v43, v44);
        v43 = (__int64)v63;
        if ( v63 == *(_QWORD ***)(v51 + 48) )
        {
          v60 = (_QWORD *)v52;
        }
        else
        {
          v53 = *(_WORD *)(v52 + 24);
          if ( v53 )
          {
            if ( v53 != 6 )
              goto LABEL_186;
            v52 = **(_QWORD **)(v52 + 32);
            if ( *(_WORD *)(v52 + 24) )
              goto LABEL_186;
          }
          v54 = *(_QWORD *)(v52 + 32);
          v73 = *(_DWORD *)(v54 + 32);
          if ( v73 > 0x40 )
            sub_C43780((__int64)&v72, (const void **)(v54 + 24));
          else
            v72 = *(_QWORD *)(v54 + 24);
          sub_9692E0((__int64)&v74, (__int64 *)&v72);
          v77 = v67;
          if ( v67 > 0x40 )
            sub_C43780((__int64)&v76, (const void **)&v66);
          else
            v76 = (unsigned __int64)v66;
          sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
          if ( v67 > 0x40 && v66 )
            j_j___libc_free_0_0((unsigned __int64)v66);
          v66 = v78;
          v67 = v79;
          if ( v77 > 0x40 && v76 )
            j_j___libc_free_0_0(v76);
          if ( v75 > 0x40 && v74 )
            j_j___libc_free_0_0(v74);
          if ( v73 > 0x40 && v72 )
            j_j___libc_free_0_0(v72);
        }
        v49 = v67;
        v51 = **(_QWORD **)(v51 + 32);
        v45 = *(__int64 **)(a1 + 8);
        continue;
      }
      break;
    }
    v55 = sub_DCC810(v45, v58, (__int64)v60, 0, 0);
    v56 = *((_WORD *)v55 + 12);
    if ( !v56 || v56 == 6 && (v55 = *(_QWORD **)v55[4], !*((_WORD *)v55 + 12)) )
    {
      sub_9865C0((__int64)&v72, v55[4] + 24LL);
      sub_9692E0((__int64)&v74, (__int64 *)&v72);
      sub_9865C0((__int64)&v76, (__int64)&v66);
      sub_C49E90((__int64)&v78, (__int64)&v76, (__int64)&v74);
      sub_228AD30((__int64)&v66, (__int64)&v78);
      sub_969240((__int64 *)&v78);
      sub_969240((__int64 *)&v76);
      sub_969240((__int64 *)&v74);
      if ( !sub_D94970((__int64)&v66, 0) )
      {
        sub_C4B8A0((__int64)&v78, (__int64)v70, (__int64)&v66);
        sub_228AD30((__int64)v71, (__int64)&v78);
        sub_969240((__int64 *)&v78);
        if ( !sub_D94970((__int64)v71, 0) )
        {
          v57 = (_BYTE *)(*(_QWORD *)(a4 + 48) + 16LL * ((unsigned int)sub_228D710(a1, v63) - 1));
          *v57 &= ~2u;
        }
      }
      sub_969240((__int64 *)&v72);
    }
    if ( *(_WORD *)(v59 + 24) == 8 )
    {
      v40 = v59;
      continue;
    }
    break;
  }
LABEL_186:
  v29 = (unsigned __int8)v29;
LABEL_187:
  sub_969240(v71);
LABEL_58:
  sub_969240(v70);
LABEL_108:
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0((unsigned __int64)v68);
LABEL_111:
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0((unsigned __int64)v66);
  return v29;
}
