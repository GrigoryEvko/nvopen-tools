// Function: sub_2DC22F0
// Address: 0x2dc22f0
//
__int64 __fastcall sub_2DC22F0(__int64 a1, __int64 **a2, __int64 **a3, __int64 **a4, unsigned int a5)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rcx
  unsigned __int8 *v9; // r13
  unsigned __int8 *v10; // r15
  __int64 **v11; // r12
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // rdi
  __int64 (__fastcall *v29)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v30; // r15
  _QWORD *v32; // rax
  __int64 v33; // rax
  char v34; // cl
  unsigned __int64 v35; // rdx
  char v36; // al
  unsigned __int64 v37; // r12
  char v38; // r12
  _QWORD *v39; // rax
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rdx
  unsigned int v44; // esi
  char v45; // r12
  _QWORD *v46; // rax
  _QWORD *v47; // r11
  __int64 v48; // rax
  __int64 v49; // r11
  __int64 v50; // r12
  __int64 v51; // r13
  __int64 v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  _QWORD *v55; // rax
  __int64 v56; // r12
  __int64 v57; // r15
  __int64 v58; // rdx
  unsigned int v59; // esi
  _QWORD *v60; // rax
  __int64 v61; // r12
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rdx
  unsigned int v65; // esi
  _QWORD *v66; // rax
  __int64 v67; // r15
  __int64 v68; // r12
  __int64 v69; // rbx
  __int64 v70; // r12
  __int64 v71; // rdx
  unsigned int v72; // esi
  _QWORD *v73; // rax
  __int64 v74; // r12
  __int64 v75; // r14
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // [rsp+10h] [rbp-D0h]
  char v79; // [rsp+20h] [rbp-C0h]
  _QWORD *v80; // [rsp+20h] [rbp-C0h]
  __int64 v81; // [rsp+20h] [rbp-C0h]
  char v82; // [rsp+28h] [rbp-B8h]
  _QWORD *v83; // [rsp+28h] [rbp-B8h]
  __int64 v84; // [rsp+28h] [rbp-B8h]
  __int64 **v86; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v87; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v88; // [rsp+48h] [rbp-98h] BYREF
  _BYTE v89[32]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v90; // [rsp+70h] [rbp-70h]
  _BYTE v91[32]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v92; // [rsp+A0h] [rbp-40h]

  v6 = a5;
  v7 = a1;
  v8 = *(_QWORD *)a1;
  v86 = a3;
  v9 = *(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
  v10 = *(unsigned __int8 **)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
  v79 = sub_BD5420(v9, *(_QWORD *)(a1 + 112));
  v82 = sub_BD5420(v10, *(_QWORD *)(a1 + 112));
  if ( (_DWORD)v6 )
  {
    v32 = (_QWORD *)sub_BD5C60(*(_QWORD *)a1);
    v92 = 257;
    v78 = sub_BCB2B0(v32);
    v9 = (unsigned __int8 *)sub_2DC20A0((__int64 *)(a1 + 128), v78, (__int64)v9, v6, (__int64)v91);
    v92 = 257;
    v33 = sub_2DC20A0((__int64 *)(a1 + 128), v78, (__int64)v10, v6, (__int64)v91);
    v34 = v79;
    v79 = -1;
    v10 = (unsigned __int8 *)v33;
    v35 = (v6 | (1LL << v34)) & -(v6 | (1LL << v34));
    if ( v35 )
    {
      _BitScanReverse64(&v35, v35);
      v79 = 63 - (v35 ^ 0x3F);
    }
    v36 = v82;
    v82 = -1;
    v37 = -(v6 | (1LL << v36)) & (v6 | (1LL << v36));
    if ( v37 )
    {
      _BitScanReverse64(&v37, v37);
      v82 = 63 - (v37 ^ 0x3F);
    }
  }
  v87 = 0;
  if ( *v9 > 0x15u || (v87 = sub_9718F0((__int64)v9, (__int64)a2, *(_BYTE **)(a1 + 112))) == 0 )
  {
    v92 = 257;
    v45 = v79;
    v90 = 257;
    v46 = sub_BD2C40(80, 1u);
    v47 = v46;
    if ( v46 )
    {
      v80 = v46;
      sub_B4D190((__int64)v46, (__int64)a2, (__int64)v9, (__int64)v91, 0, v45, 0, 0);
      v47 = v80;
    }
    v81 = (__int64)v47;
    (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 216) + 16LL))(
      *(_QWORD *)(a1 + 216),
      v47,
      v89,
      *(_QWORD *)(a1 + 184),
      *(_QWORD *)(a1 + 192));
    v48 = *(_QWORD *)(a1 + 128);
    v49 = v81;
    v50 = 16LL * *(unsigned int *)(a1 + 136);
    v51 = v48 + v50;
    if ( v48 != v48 + v50 )
    {
      v52 = *(_QWORD *)(a1 + 128);
      do
      {
        v53 = *(_QWORD *)(v52 + 8);
        v54 = *(_DWORD *)v52;
        v52 += 16;
        sub_B99FD0(v81, v54, v53);
      }
      while ( v51 != v52 );
      v7 = a1;
      v49 = v81;
    }
    v87 = v49;
  }
  v88 = 0;
  if ( *v10 > 0x15u || (v88 = sub_9718F0((__int64)v10, (__int64)a2, *(_BYTE **)(v7 + 112))) == 0 )
  {
    v90 = 257;
    v38 = v82;
    v92 = 257;
    v39 = sub_BD2C40(80, 1u);
    if ( v39 )
    {
      v83 = v39;
      sub_B4D190((__int64)v39, (__int64)a2, (__int64)v10, (__int64)v91, 0, v38, 0, 0);
      v39 = v83;
    }
    v84 = (__int64)v39;
    (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 216) + 16LL))(
      *(_QWORD *)(v7 + 216),
      v39,
      v89,
      *(_QWORD *)(v7 + 184),
      *(_QWORD *)(v7 + 192));
    v40 = *(_QWORD *)(v7 + 128);
    v41 = v84;
    v42 = v40 + 16LL * *(unsigned int *)(v7 + 136);
    if ( v40 != v42 )
    {
      do
      {
        v43 = *(_QWORD *)(v40 + 8);
        v44 = *(_DWORD *)v40;
        v40 += 16;
        sub_B99FD0(v84, v44, v43);
      }
      while ( v42 != v40 );
      v41 = v84;
    }
    v88 = v41;
  }
  v11 = v86;
  if ( v86 )
  {
    if ( v86 == a2 )
      goto LABEL_23;
    v90 = 257;
    if ( v86 == *(__int64 ***)(v87 + 8) )
    {
      v14 = v87;
LABEL_15:
      v87 = v14;
      v90 = 257;
      if ( *(__int64 ***)(v88 + 8) == v11 )
      {
        v17 = v88;
        goto LABEL_22;
      }
      v15 = *(_QWORD *)(v7 + 208);
      v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v15 + 120LL);
      if ( v16 == sub_920130 )
      {
        if ( *(_BYTE *)v88 > 0x15u )
          goto LABEL_76;
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v17 = sub_ADAB70(39, v88, v11, 0);
        else
          v17 = sub_AA93C0(0x27u, v88, (__int64)v11);
      }
      else
      {
        v17 = v16(v15, 39u, (_BYTE *)v88, (__int64)v11);
      }
      if ( v17 )
      {
LABEL_21:
        v11 = v86;
LABEL_22:
        v88 = v17;
LABEL_23:
        if ( v11 )
        {
          v18 = (__int64 *)sub_B43CA0(*(_QWORD *)v7);
          v19 = sub_B6E160(v18, 0xFu, (__int64)&v86, 1);
          v92 = 257;
          v20 = v19;
          if ( v19 )
          {
            v21 = sub_921880((unsigned int **)(v7 + 128), *(_QWORD *)(v19 + 24), v19, (int)&v87, 1, (__int64)v91, 0);
            v92 = 257;
            v22 = *(_QWORD *)(v20 + 24);
            v87 = v21;
          }
          else
          {
            v22 = 0;
            v87 = sub_921880((unsigned int **)(v7 + 128), 0, 0, (int)&v87, 1, (__int64)v91, 0);
            v92 = 257;
          }
          v88 = sub_921880((unsigned int **)(v7 + 128), v22, v20, (int)&v88, 1, (__int64)v91, 0);
        }
        goto LABEL_27;
      }
LABEL_76:
      v92 = 257;
      v73 = sub_BD2C40(72, 1u);
      v17 = (__int64)v73;
      if ( v73 )
        sub_B515B0((__int64)v73, v88, (__int64)v11, (__int64)v91, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 216) + 16LL))(
        *(_QWORD *)(v7 + 216),
        v17,
        v89,
        *(_QWORD *)(v7 + 184),
        *(_QWORD *)(v7 + 192));
      v74 = *(_QWORD *)(v7 + 128);
      v75 = v74 + 16LL * *(unsigned int *)(v7 + 136);
      while ( v75 != v74 )
      {
        v76 = *(_QWORD *)(v74 + 8);
        v77 = *(_DWORD *)v74;
        v74 += 16;
        sub_B99FD0(v17, v77, v76);
      }
      goto LABEL_21;
    }
    v12 = *(_QWORD *)(v7 + 208);
    v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v12 + 120LL);
    if ( v13 == sub_920130 )
    {
      if ( *(_BYTE *)v87 > 0x15u )
        goto LABEL_61;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v14 = sub_ADAB70(39, v87, v86, 0);
      else
        v14 = sub_AA93C0(0x27u, v87, (__int64)v86);
    }
    else
    {
      v14 = v13(v12, 39u, (_BYTE *)v87, (__int64)v86);
    }
    if ( v14 )
    {
LABEL_14:
      v11 = v86;
      goto LABEL_15;
    }
LABEL_61:
    v92 = 257;
    v55 = sub_BD2C40(72, 1u);
    v14 = (__int64)v55;
    if ( v55 )
      sub_B515B0((__int64)v55, v87, (__int64)v86, (__int64)v91, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 216) + 16LL))(
      *(_QWORD *)(v7 + 216),
      v14,
      v89,
      *(_QWORD *)(v7 + 184),
      *(_QWORD *)(v7 + 192));
    v56 = *(_QWORD *)(v7 + 128);
    v57 = v56 + 16LL * *(unsigned int *)(v7 + 136);
    while ( v57 != v56 )
    {
      v58 = *(_QWORD *)(v56 + 8);
      v59 = *(_DWORD *)v56;
      v56 += 16;
      sub_B99FD0(v14, v59, v58);
    }
    goto LABEL_14;
  }
LABEL_27:
  v23 = v87;
  if ( a4 && a4 != *(__int64 ***)(v87 + 8) )
  {
    v90 = 257;
    if ( a4 == *(__int64 ***)(v87 + 8) )
    {
      v26 = v87;
      goto LABEL_35;
    }
    v24 = *(_QWORD *)(v7 + 208);
    v25 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v24 + 120LL);
    if ( v25 == sub_920130 )
    {
      if ( *(_BYTE *)v87 > 0x15u )
      {
LABEL_66:
        v92 = 257;
        v60 = sub_BD2C40(72, 1u);
        v26 = (__int64)v60;
        if ( v60 )
          sub_B515B0((__int64)v60, v23, (__int64)a4, (__int64)v91, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 216) + 16LL))(
          *(_QWORD *)(v7 + 216),
          v26,
          v89,
          *(_QWORD *)(v7 + 184),
          *(_QWORD *)(v7 + 192));
        v61 = *(_QWORD *)(v7 + 128);
        v62 = 16LL * *(unsigned int *)(v7 + 136);
        v63 = v61 + v62;
        while ( v63 != v61 )
        {
          v64 = *(_QWORD *)(v61 + 8);
          v65 = *(_DWORD *)v61;
          v61 += 16;
          sub_B99FD0(v26, v65, v64);
        }
LABEL_35:
        v27 = v88;
        v87 = v26;
        v90 = 257;
        if ( a4 == *(__int64 ***)(v88 + 8) )
          return v26;
        v28 = *(_QWORD *)(v7 + 208);
        v29 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v28 + 120LL);
        if ( v29 == sub_920130 )
        {
          if ( *(_BYTE *)v88 > 0x15u )
            goto LABEL_71;
          if ( (unsigned __int8)sub_AC4810(0x27u) )
            v30 = sub_ADAB70(39, v27, a4, 0);
          else
            v30 = sub_AA93C0(0x27u, v27, (__int64)a4);
        }
        else
        {
          v30 = v29(v28, 39u, (_BYTE *)v88, (__int64)a4);
        }
        if ( v30 )
          return v87;
LABEL_71:
        v92 = 257;
        v66 = sub_BD2C40(72, 1u);
        v67 = (__int64)v66;
        if ( v66 )
          sub_B515B0((__int64)v66, v27, (__int64)a4, (__int64)v91, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 216) + 16LL))(
          *(_QWORD *)(v7 + 216),
          v67,
          v89,
          *(_QWORD *)(v7 + 184),
          *(_QWORD *)(v7 + 192));
        v68 = 16LL * *(unsigned int *)(v7 + 136);
        v69 = *(_QWORD *)(v7 + 128);
        v70 = v69 + v68;
        while ( v70 != v69 )
        {
          v71 = *(_QWORD *)(v69 + 8);
          v72 = *(_DWORD *)v69;
          v69 += 16;
          sub_B99FD0(v67, v72, v71);
        }
        return v87;
      }
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v26 = sub_ADAB70(39, v23, a4, 0);
      else
        v26 = sub_AA93C0(0x27u, v23, (__int64)a4);
    }
    else
    {
      v26 = v25(v24, 39u, (_BYTE *)v87, (__int64)a4);
    }
    if ( v26 )
      goto LABEL_35;
    goto LABEL_66;
  }
  return v23;
}
