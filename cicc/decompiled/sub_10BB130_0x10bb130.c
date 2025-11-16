// Function: sub_10BB130
// Address: 0x10bb130
//
__int64 __fastcall sub_10BB130(__int64 a1, int a2, char a3)
{
  unsigned int v4; // r13d
  __int64 v5; // r12
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  __int64 **v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rsi
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 **v19; // rax
  __int64 *v20; // r15
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // r12
  __int64 v24; // r12
  __int64 v25; // r12
  bool v26; // r15
  __int64 v27; // r12
  unsigned int v29; // r15d
  __int64 **v30; // r15
  __int64 **v31; // r12
  __int64 *v32; // r14
  __int64 *v33; // r8
  char v34; // al
  __int64 *v35; // rsi
  __int64 *v36; // r14
  unsigned int v37; // eax
  unsigned int v38; // r12d
  __int64 v39; // r14
  __int64 v40; // r14
  __int64 *v41; // rax
  __int64 *v42; // r15
  __int64 v43; // r12
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 *v46; // rbx
  __int64 v47; // r15
  const void **v48; // rsi
  __int64 *v49; // rsi
  __int64 *v50; // r15
  unsigned int v51; // r12d
  __int64 v52; // r14
  __int64 v53; // r14
  _QWORD **v54; // rdx
  int v55; // ecx
  int v56; // eax
  __int64 *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r13
  __int64 v60; // rbx
  __int64 v61; // r13
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // r12
  __int64 v65; // r15
  __int64 v66; // rdx
  unsigned int v67; // esi
  unsigned int v68; // r12d
  __int64 v69; // r14
  __int64 *v70; // [rsp+8h] [rbp-E8h]
  __int64 v71; // [rsp+8h] [rbp-E8h]
  __int64 v72; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v73; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v74; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v75; // [rsp+28h] [rbp-C8h]
  __int64 v76; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v77; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v78; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v79; // [rsp+48h] [rbp-A8h]
  __int64 v80; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v81; // [rsp+58h] [rbp-98h]
  __int64 v82; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v83; // [rsp+68h] [rbp-88h]
  __int16 v84; // [rsp+80h] [rbp-70h]
  unsigned __int64 v85; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v86; // [rsp+98h] [rbp-58h]
  __int16 v87; // [rsp+B0h] [rbp-40h]

  v4 = a2;
  if ( a3 )
    v4 = sub_B52870(a2);
  v5 = **(_QWORD **)(a1 + 16);
  if ( **(_DWORD **)a1 != v4 )
  {
    v6 = **(_QWORD **)(a1 + 8);
    v7 = *(_DWORD *)(v6 + 8);
    v86 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43780((__int64)&v85, (const void **)v6);
      v7 = v86;
      if ( v86 > 0x40 )
      {
        sub_C43C10(&v85, (__int64 *)v5);
        v9 = v85;
        v7 = v86;
        goto LABEL_7;
      }
      v8 = v85;
    }
    else
    {
      v8 = *(_QWORD *)v6;
      v85 = *(_QWORD *)v6;
    }
    v9 = *(_QWORD *)v5 ^ v8;
LABEL_7:
    v73 = v7;
    v72 = v9;
    goto LABEL_8;
  }
  v73 = *(_DWORD *)(v5 + 8);
  if ( v73 > 0x40 )
    sub_C43780((__int64)&v72, (const void **)v5);
  else
    v72 = *(_QWORD *)v5;
LABEL_8:
  v10 = *(__int64 ***)(a1 + 40);
  if ( **(_DWORD **)(a1 + 24) != v4 )
  {
    v11 = *v10;
    v12 = **(_QWORD **)(a1 + 32);
    v13 = *(_DWORD *)(v12 + 8);
    v86 = v13;
    if ( v13 > 0x40 )
    {
      sub_C43780((__int64)&v85, (const void **)v12);
      v13 = v86;
      if ( v86 > 0x40 )
      {
        sub_C43C10(&v85, v11);
        v13 = v86;
        v15 = v85;
        goto LABEL_12;
      }
      v14 = v85;
    }
    else
    {
      v14 = *(_QWORD *)v12;
      v85 = *(_QWORD *)v12;
    }
    v15 = *v11 ^ v14;
LABEL_12:
    v75 = v13;
    v74 = v15;
    goto LABEL_13;
  }
  v48 = (const void **)*v10;
  v75 = *((_DWORD *)*v10 + 2);
  if ( v75 > 0x40 )
    sub_C43780((__int64)&v74, v48);
  else
    v74 = (unsigned __int64)*v48;
LABEL_13:
  v16 = v73;
  v81 = v73;
  if ( v73 <= 0x40 )
  {
    v17 = v72;
LABEL_15:
    v18 = v74 ^ v17;
    v80 = v18;
    goto LABEL_16;
  }
  sub_C43780((__int64)&v80, (const void **)&v72);
  v16 = v81;
  if ( v81 <= 0x40 )
  {
    v17 = v80;
    goto LABEL_15;
  }
  sub_C43C10(&v80, (__int64 *)&v74);
  v16 = v81;
  v18 = v80;
LABEL_16:
  v82 = v18;
  v19 = *(__int64 ***)(a1 + 32);
  v83 = v16;
  v81 = 0;
  v20 = *v19;
  v21 = **(_QWORD **)(a1 + 8);
  v22 = *(_DWORD *)(v21 + 8);
  v77 = v22;
  if ( v22 <= 0x40 )
  {
    v23 = *(_QWORD *)v21;
    v76 = *(_QWORD *)v21;
LABEL_18:
    v24 = *v20 & v23;
    v76 = v24;
    goto LABEL_19;
  }
  sub_C43780((__int64)&v76, (const void **)v21);
  v22 = v77;
  if ( v77 <= 0x40 )
  {
    v23 = v76;
    v16 = v83;
    goto LABEL_18;
  }
  sub_C43B90(&v76, v20);
  v22 = v77;
  v24 = v76;
  v16 = v83;
LABEL_19:
  v79 = v22;
  v78 = v24;
  v77 = 0;
  if ( v16 <= 0x40 )
  {
    v25 = v82 & v24;
    v83 = 0;
    v82 = v25;
LABEL_21:
    v26 = v25 == 0;
    goto LABEL_22;
  }
  sub_C43B90(&v82, (__int64 *)&v78);
  v29 = v83;
  v25 = v82;
  v83 = 0;
  v86 = v29;
  v85 = v82;
  if ( v29 <= 0x40 )
  {
    v22 = v79;
    goto LABEL_21;
  }
  v26 = v29 == (unsigned int)sub_C444A0((__int64)&v85);
  if ( v25 )
    j_j___libc_free_0_0(v25);
  v22 = v79;
LABEL_22:
  if ( v22 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( !v26 )
  {
    v27 = 0;
    if ( !a3 )
      v27 = sub_AD64C0(*(_QWORD *)(**(_QWORD **)(a1 + 48) + 8LL), **(_BYTE **)(a1 + 56) ^ 1u, 0);
    goto LABEL_37;
  }
  v30 = *(__int64 ***)(a1 + 32);
  v31 = *(__int64 ***)(a1 + 8);
  if ( a3 )
  {
    v32 = *v31;
    v33 = *v30;
    if ( *((_DWORD *)*v31 + 2) <= 0x40u )
    {
      if ( (*v32 & ~*v33) == 0 )
        goto LABEL_55;
    }
    else
    {
      v70 = *v30;
      v34 = sub_C446F0(*v31, *v30);
      v33 = v70;
      if ( v34 )
      {
LABEL_55:
        v79 = 1;
        v78 = 0;
        v81 = 1;
        v80 = 0;
        v35 = *v31;
        v36 = *v30;
        v37 = *((_DWORD *)*v31 + 2);
        v86 = v37;
        if ( v37 > 0x40 )
        {
          sub_C43780((__int64)&v85, (const void **)v35);
          v68 = v86;
          if ( v86 <= 0x40 )
          {
            v69 = v85 & *v36;
            v85 = v69;
          }
          else
          {
            sub_C43B90(&v85, v36);
            v68 = v86;
            v69 = v85;
          }
          v86 = 0;
          if ( v79 > 0x40 && v78 )
          {
            j_j___libc_free_0_0(v78);
            v78 = v69;
            v79 = v68;
            if ( v86 > 0x40 && v85 )
              j_j___libc_free_0_0(v85);
          }
          else
          {
            v78 = v69;
            v79 = v68;
          }
        }
        else
        {
          v85 = *v35;
          v78 = *v36 & v85;
          v79 = v37;
        }
        v38 = v73;
        v86 = v73;
        if ( v73 <= 0x40 )
        {
          v39 = v72;
LABEL_59:
          v40 = v74 & v39;
          v85 = v40;
          goto LABEL_60;
        }
        sub_C43780((__int64)&v85, (const void **)&v72);
        v38 = v86;
        if ( v86 <= 0x40 )
        {
          v39 = v85;
          goto LABEL_59;
        }
        sub_C43B90(&v85, (__int64 *)&v74);
        v38 = v86;
        v40 = v85;
LABEL_60:
        v86 = 0;
        if ( v81 <= 0x40 )
          goto LABEL_87;
        goto LABEL_61;
      }
    }
    if ( *((_DWORD *)v33 + 2) <= 0x40u )
    {
      if ( (*v33 & ~*v32) == 0 )
        goto LABEL_55;
    }
    else if ( (unsigned __int8)sub_C446F0(v33, v32) )
    {
      goto LABEL_55;
    }
    v27 = 0;
    goto LABEL_37;
  }
  v79 = 1;
  v78 = 0;
  v81 = 1;
  v80 = 0;
  v49 = *v31;
  v50 = *v30;
  v51 = *((_DWORD *)*v31 + 2);
  v86 = v51;
  if ( v51 <= 0x40 )
  {
    v85 = *v49;
    v52 = *v50 | v85;
LABEL_83:
    v78 = v52;
    v79 = v51;
    goto LABEL_84;
  }
  sub_C43780((__int64)&v85, (const void **)v49);
  v51 = v86;
  if ( v86 <= 0x40 )
  {
    v52 = v85 | *v50;
    v85 = v52;
  }
  else
  {
    sub_C43BD0(&v85, v50);
    v51 = v86;
    v52 = v85;
  }
  v86 = 0;
  if ( v79 <= 0x40 || !v78 )
    goto LABEL_83;
  j_j___libc_free_0_0(v78);
  v78 = v52;
  v79 = v51;
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
LABEL_84:
  v38 = v73;
  v86 = v73;
  if ( v73 > 0x40 )
  {
    sub_C43780((__int64)&v85, (const void **)&v72);
    v38 = v86;
    if ( v86 > 0x40 )
    {
      sub_C43BD0(&v85, (__int64 *)&v74);
      v38 = v86;
      v40 = v85;
      goto LABEL_60;
    }
    v53 = v85;
  }
  else
  {
    v53 = v72;
  }
  v40 = v74 | v53;
  v86 = 0;
  v85 = v40;
  if ( v81 <= 0x40 )
    goto LABEL_87;
LABEL_61:
  if ( v80 )
  {
    j_j___libc_free_0_0(v80);
    v80 = v40;
    v81 = v38;
    if ( v86 > 0x40 && v85 )
      j_j___libc_free_0_0(v85);
    goto LABEL_65;
  }
LABEL_87:
  v80 = v40;
  v81 = v38;
LABEL_65:
  v41 = *(__int64 **)(a1 + 72);
  v42 = *(__int64 **)(a1 + 64);
  v84 = 257;
  v43 = *v41;
  v71 = sub_AD8D80(*(_QWORD *)(*v41 + 8), (__int64)&v78);
  v44 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v42[10] + 16LL))(
          v42[10],
          28,
          v43,
          v71);
  if ( !v44 )
  {
    v87 = 257;
    v44 = sub_B504D0(28, v43, v71, (__int64)&v85, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v42[11] + 16LL))(
      v42[11],
      v44,
      &v82,
      v42[7],
      v42[8]);
    v64 = *v42;
    v65 = *v42 + 16LL * *((unsigned int *)v42 + 2);
    while ( v65 != v64 )
    {
      v66 = *(_QWORD *)(v64 + 8);
      v67 = *(_DWORD *)v64;
      v64 += 16;
      sub_B99FD0(v44, v67, v66);
    }
  }
  v45 = sub_AD8D80(*(_QWORD *)(**(_QWORD **)(a1 + 72) + 8LL), (__int64)&v80);
  v46 = *(__int64 **)(a1 + 64);
  v47 = v45;
  v84 = 257;
  v27 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v46[10] + 56LL))(
          v46[10],
          v4,
          v45,
          v44);
  if ( !v27 )
  {
    v87 = 257;
    v27 = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( v27 )
    {
      v54 = *(_QWORD ***)(v47 + 8);
      v55 = *((unsigned __int8 *)v54 + 8);
      if ( (unsigned int)(v55 - 17) > 1 )
      {
        v58 = sub_BCB2A0(*v54);
      }
      else
      {
        v56 = *((_DWORD *)v54 + 8);
        BYTE4(v76) = (_BYTE)v55 == 18;
        LODWORD(v76) = v56;
        v57 = (__int64 *)sub_BCB2A0(*v54);
        v58 = sub_BCE1B0(v57, v76);
      }
      sub_B523C0(v27, v58, 53, v4, v47, v44, (__int64)&v85, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v46[11] + 16LL))(
      v46[11],
      v27,
      &v82,
      v46[7],
      v46[8]);
    v59 = 16LL * *((unsigned int *)v46 + 2);
    v60 = *v46;
    v61 = v60 + v59;
    while ( v61 != v60 )
    {
      v62 = *(_QWORD *)(v60 + 8);
      v63 = *(_DWORD *)v60;
      v60 += 16;
      sub_B99FD0(v27, v63, v62);
    }
  }
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
LABEL_37:
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  return v27;
}
