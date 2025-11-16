// Function: sub_D30800
// Address: 0xd30800
//
__int64 __fastcall sub_D30800(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned int v15; // eax
  unsigned int v16; // r10d
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  unsigned int v22; // eax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  unsigned int v28; // edx
  unsigned __int8 v29; // r10
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // r14
  char v37; // al
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  __int16 v41; // ax
  char v42; // r9
  unsigned __int8 *v43; // rbx
  __int64 v44; // rax
  __int64 v45; // r8
  const void **v46; // rsi
  _QWORD *v47; // rax
  __int64 v48; // r8
  __int64 v49; // rbx
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 v52; // rdi
  unsigned __int64 v53; // rax
  __int64 v54; // r14
  __int64 *v55; // r14
  unsigned int v56; // eax
  unsigned int v57; // eax
  unsigned __int8 v58; // [rsp+8h] [rbp-B8h]
  __int64 v59; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v60; // [rsp+10h] [rbp-B0h]
  unsigned int v61; // [rsp+10h] [rbp-B0h]
  int v62; // [rsp+10h] [rbp-B0h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v66; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v69; // [rsp+34h] [rbp-8Ch]
  unsigned __int8 v70; // [rsp+38h] [rbp-88h]
  unsigned __int8 v71; // [rsp+38h] [rbp-88h]
  unsigned __int64 v72; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v73; // [rsp+48h] [rbp-78h]
  const void *v74; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v75; // [rsp+58h] [rbp-68h]
  const void *v76; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v77; // [rsp+68h] [rbp-58h]
  const void *v78; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v79; // [rsp+78h] [rbp-48h]
  const void *v80; // [rsp+80h] [rbp-40h] BYREF
  __int64 v81; // [rsp+88h] [rbp-38h]

  _BitScanReverse64(&v9, 1LL << (*(_WORD *)(a1 + 2) >> 1));
  v69 = 63 - (v9 ^ 0x3F);
  v10 = sub_B43CC0(a1);
  v11 = *(_QWORD *)(a1 - 32);
  v12 = v10;
  v80 = (const void *)sub_9208B0(v10, *(_QWORD *)(a1 + 8));
  v81 = v13;
  v73 = sub_AE43F0(v12, *(_QWORD *)(v11 + 8));
  if ( v73 > 0x40 )
    sub_C43690((__int64)&v72, ((unsigned __int64)v80 + 7) >> 3, 0);
  else
    v72 = ((unsigned __int64)v80 + 7) >> 3;
  if ( !(unsigned __int8)sub_D48480(a2, v11) )
  {
    v18 = sub_DD8400(a3, v11);
    v16 = 0;
    v19 = v18;
    if ( *(_WORD *)(v18 + 24) != 8 )
      goto LABEL_7;
    if ( a2 != *(_QWORD *)(v18 + 48) )
      goto LABEL_7;
    if ( *(_QWORD *)(v18 + 40) != 2 )
      goto LABEL_7;
    v59 = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
    if ( *(_WORD *)(v59 + 24) )
      goto LABEL_7;
    v20 = sub_C459C0((__int64)&v72, 1LL << v69);
    v16 = 0;
    if ( v20 )
      goto LABEL_7;
    v21 = *(_QWORD *)(v59 + 32);
    v22 = *(_DWORD *)(v21 + 32);
    v23 = *(_QWORD *)(v21 + 24);
    v24 = 1LL << ((unsigned __int8)v22 - 1);
    if ( v22 > 0x40 )
    {
      v46 = (const void **)(v21 + 24);
      if ( (*(_QWORD *)(v23 + 8LL * ((v22 - 1) >> 6)) & v24) == 0 )
      {
        v79 = v22;
        sub_C43780((__int64)&v78, v46);
        v28 = v79;
        v29 = 0;
        goto LABEL_23;
      }
      LODWORD(v81) = v22;
      sub_C43780((__int64)&v80, v46);
      v22 = v81;
      LOBYTE(v16) = 0;
      if ( (unsigned int)v81 > 0x40 )
      {
        sub_C43D10((__int64)&v80);
        LOBYTE(v16) = 0;
LABEL_22:
        v60 = v16;
        sub_C46250((__int64)&v80);
        v28 = v81;
        v29 = v60;
        v78 = v80;
        v79 = v81;
LABEL_23:
        v58 = v29;
        v61 = v28;
        v30 = sub_C49970((__int64)&v72, (unsigned __int64 *)&v78);
        v16 = v58;
        if ( v61 > 0x40 && v78 )
        {
          v62 = v30;
          j_j___libc_free_0_0(v78);
          v16 = v58;
          v30 = v62;
        }
LABEL_26:
        if ( v30 > 0 )
          goto LABEL_7;
        v31 = a6;
        v66 = v16;
        v32 = v31 ? sub_DBB040(a3, a2) : sub_DCF3A0(a3, a2, 1);
        v63 = v32;
        v33 = sub_D96A50(v32);
        v16 = v66;
        if ( v33 )
          goto LABEL_7;
        v34 = sub_D3B9E0(a2, v19, *(_QWORD *)(a1 + 8), v63, a3, 0);
        v36 = v35;
        v37 = sub_D96A50(v34);
        v16 = v66;
        if ( v37 )
          goto LABEL_7;
        v38 = sub_D96A50(v36);
        v16 = v66;
        if ( v38 )
          goto LABEL_7;
        v39 = sub_DCC810(a3, v36, v34, 0, 0);
        v40 = sub_DBB9F0(a3, v39, 0, 0);
        sub_AB0910((__int64)&v74, v40);
        v41 = *(_WORD *)(v34 + 24);
        v77 = 1;
        v76 = 0;
        v42 = 0;
        if ( v41 == 15 )
        {
          v43 = *(unsigned __int8 **)(v34 - 8);
          if ( v75 <= 0x40 )
          {
            v77 = v75;
            v76 = v74;
          }
          else
          {
            sub_C43990((__int64)&v76, (__int64)&v74);
          }
        }
        else
        {
          if ( v41 != 5 )
            goto LABEL_38;
          if ( *(_QWORD *)(v34 + 40) != 2 )
            goto LABEL_38;
          v47 = *(_QWORD **)(v34 + 32);
          v48 = *v47;
          if ( *(_WORD *)(*v47 + 24LL) )
            goto LABEL_38;
          v49 = v47[1];
          if ( *(_WORD *)(v49 + 24) != 15 )
            goto LABEL_38;
          v50 = *(_QWORD *)(v48 + 32);
          v51 = *(_DWORD *)(v50 + 32);
          v52 = *(_QWORD *)(v50 + 24);
          if ( v51 > 0x40 )
            v52 = *(_QWORD *)(v52 + 8LL * ((v51 - 1) >> 6));
          v64 = v48;
          if ( (v52 & (1LL << ((unsigned __int8)v51 - 1))) != 0 )
            goto LABEL_38;
          v53 = sub_C459C0(v50 + 24, 1LL << v69);
          v42 = 0;
          if ( v53 )
            goto LABEL_38;
          v54 = *(_QWORD *)(v64 + 32);
          v79 = v75;
          v55 = (__int64 *)(v54 + 24);
          if ( v75 > 0x40 )
            sub_C43780((__int64)&v78, &v74);
          else
            v78 = v74;
          sub_C45EE0((__int64)&v78, v55);
          v56 = v79;
          v79 = 0;
          LODWORD(v81) = v56;
          v80 = v78;
          if ( v77 > 0x40 && v76 )
            j_j___libc_free_0_0(v76);
          v76 = v80;
          v57 = v81;
          LODWORD(v81) = 0;
          v77 = v57;
          sub_969240((__int64 *)&v80);
          sub_969240((__int64 *)&v78);
          v43 = *(unsigned __int8 **)(v49 - 8);
        }
        v44 = sub_AA4FF0(**(_QWORD **)(a2 + 32));
        v45 = v44;
        if ( v44 )
          v45 = v44 - 24;
        v42 = sub_D30550(v43, v69, (unsigned __int64 *)&v76, v12, v45, a5, a4, 0);
LABEL_38:
        v71 = v42;
        sub_969240((__int64 *)&v76);
        sub_969240((__int64 *)&v74);
        v16 = v71;
        goto LABEL_7;
      }
      v25 = (__int64)v80;
    }
    else
    {
      v25 = *(_QWORD *)(v21 + 24);
      if ( (v24 & v23) == 0 )
      {
        v79 = *(_DWORD *)(v21 + 32);
        v78 = (const void *)v23;
        v30 = sub_C49970((__int64)&v72, (unsigned __int64 *)&v78);
        v16 = 0;
        goto LABEL_26;
      }
      LODWORD(v81) = *(_DWORD *)(v21 + 32);
    }
    v26 = ~v25;
    v27 = 0;
    if ( v22 )
      v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
    v80 = (const void *)(v26 & v27);
    goto LABEL_22;
  }
  v14 = sub_AA4FF0(**(_QWORD **)(a2 + 32));
  if ( v14 )
    v14 -= 24;
  LOBYTE(v15) = sub_D30550((unsigned __int8 *)v11, v69, &v72, v12, v14, a5, a4, 0);
  v16 = v15;
LABEL_7:
  if ( v73 > 0x40 && v72 )
  {
    v70 = v16;
    j_j___libc_free_0_0(v72);
    return v70;
  }
  return v16;
}
