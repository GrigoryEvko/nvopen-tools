// Function: sub_117CDE0
// Address: 0x117cde0
//
__int64 __fastcall sub_117CDE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // r12
  unsigned __int8 v8; // dl
  int v9; // eax
  char v10; // dl
  __int64 v11; // r14
  __int16 v12; // r14
  __int64 v13; // rdx
  unsigned int v14; // r14d
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // r8d
  _BYTE *v22; // rax
  int v23; // eax
  __int64 v24; // rcx
  int v25; // eax
  int v26; // eax
  char v27; // cl
  unsigned int v28; // r15d
  unsigned __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rsi
  __int64 v32; // rax
  _BYTE *v33; // rax
  __int64 v34; // rax
  int v35; // eax
  unsigned int v36; // eax
  unsigned __int64 v37; // rax
  unsigned int v38; // r13d
  int v39; // r15d
  int v40; // eax
  __int64 v41; // rcx
  bool v42; // dl
  int v43; // eax
  __int64 v44; // r13
  __int64 v45; // rsi
  unsigned int v46; // eax
  int v47; // eax
  unsigned int v48; // eax
  _BYTE *v49; // rax
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // r13
  unsigned int *v53; // r12
  __int64 v54; // rbx
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // r13
  __int64 v58; // r12
  __int64 v59; // r15
  __int64 v60; // rax
  __int64 v61; // rdi
  unsigned int v62; // r12d
  _BYTE *v63; // r14
  __int64 v64; // rax
  unsigned int *v65; // r13
  __int64 v66; // rbx
  __int64 v67; // rax
  int v68; // eax
  unsigned __int64 v69; // rax
  unsigned int *v70; // rbx
  __int64 v71; // r13
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rax
  __int64 v75; // rdx
  int v76; // r12d
  unsigned int *v77; // rbx
  __int64 v78; // r12
  __int64 v79; // rdx
  unsigned int v80; // esi
  unsigned int v81; // [rsp+Ch] [rbp-F4h]
  unsigned int v82; // [rsp+10h] [rbp-F0h]
  unsigned int v83; // [rsp+14h] [rbp-ECh]
  unsigned int v84; // [rsp+18h] [rbp-E8h]
  char v85; // [rsp+18h] [rbp-E8h]
  unsigned int v86; // [rsp+1Ch] [rbp-E4h]
  unsigned int v87; // [rsp+1Ch] [rbp-E4h]
  unsigned int v88; // [rsp+1Ch] [rbp-E4h]
  unsigned int v89; // [rsp+1Ch] [rbp-E4h]
  unsigned int v90; // [rsp+1Ch] [rbp-E4h]
  unsigned int v91; // [rsp+1Ch] [rbp-E4h]
  unsigned int v92; // [rsp+1Ch] [rbp-E4h]
  bool v93; // [rsp+23h] [rbp-DDh]
  unsigned int v94; // [rsp+24h] [rbp-DCh]
  unsigned __int8 v95; // [rsp+28h] [rbp-D8h]
  unsigned int v96; // [rsp+28h] [rbp-D8h]
  __int64 v97; // [rsp+30h] [rbp-D0h]
  __int64 v98; // [rsp+38h] [rbp-C8h]
  __int64 v99; // [rsp+40h] [rbp-C0h]
  unsigned int v100; // [rsp+40h] [rbp-C0h]
  __int64 v101; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v103; // [rsp+50h] [rbp-B0h] BYREF
  int v104; // [rsp+58h] [rbp-A8h]
  _BYTE v105[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v106; // [rsp+80h] [rbp-80h]
  __int64 v107; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v108; // [rsp+98h] [rbp-68h]
  unsigned __int64 v109; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v110; // [rsp+A8h] [rbp-58h]
  __int64 v111[2]; // [rsp+B0h] [rbp-50h] BYREF
  unsigned __int8 v112; // [rsp+C0h] [rbp-40h]

  v5 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)v5 == 17 )
  {
    v99 = v5 + 24;
  }
  else
  {
    v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v17 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v22 = sub_AD7630(v5, 0, v17);
    if ( !v22 || *v22 != 17 )
      return 0;
    v99 = (__int64)(v22 + 24);
  }
  v6 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v6 == 17 )
  {
    v98 = v6 + 24;
    goto LABEL_5;
  }
  v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
  if ( (unsigned int)v15 > 1 )
    return 0;
  if ( *(_BYTE *)v6 > 0x15u )
    return 0;
  v16 = sub_AD7630(v6, 0, v15);
  if ( !v16 || *v16 != 17 )
    return 0;
  v98 = (__int64)(v16 + 24);
LABEL_5:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
  {
    v9 = v8;
    v11 = 0;
    if ( v8 == 18 )
      return v11;
    v10 = 0;
  }
  else
  {
    v9 = v8;
    if ( v8 == 18 )
      goto LABEL_9;
    v10 = 1;
  }
  v11 = 0;
  if ( (v9 == 17) != v10 )
    return v11;
LABEL_9:
  v12 = *(_WORD *)(a2 + 2);
  v13 = *(_QWORD *)(a2 - 32);
  v104 = 1;
  v103 = 0;
  v14 = v12 & 0x3F;
  if ( v14 - 32 > 1 )
  {
    sub_11FB020(&v107, *(_QWORD *)(a2 - 64), v13, v14, 1, 0);
    v95 = v112;
    if ( v112 )
    {
      if ( v110 > 0x40 )
      {
        v23 = sub_C44630((__int64)&v109);
        v97 = v107;
        if ( v23 == 1 )
        {
          sub_C43990((__int64)&v103, (__int64)&v109);
          v14 = v108;
          if ( !v112 )
            goto LABEL_39;
          goto LABEL_15;
        }
      }
      else if ( v109 && (v109 & (v109 - 1)) == 0 )
      {
        v97 = v107;
        v103 = v109;
        v14 = v108;
        v104 = v110;
LABEL_15:
        v112 = 0;
        sub_969240(v111);
        sub_969240((__int64 *)&v109);
LABEL_39:
        v21 = v104;
        goto LABEL_40;
      }
      v112 = 0;
      sub_969240(v111);
      sub_969240((__int64 *)&v109);
    }
LABEL_27:
    v21 = v104;
    v11 = 0;
    goto LABEL_28;
  }
  if ( !(unsigned __int8)sub_1178DE0(v13) )
    goto LABEL_27;
  v20 = *(_BYTE **)(a2 - 64);
  v97 = (__int64)v20;
  if ( *v20 != 57 )
    goto LABEL_27;
  v30 = *((_QWORD *)v20 - 4);
  if ( *(_BYTE *)v30 == 17 )
  {
    v31 = v30 + 24;
    if ( *(_DWORD *)(v30 + 32) > 0x40u )
    {
      v31 = v30 + 24;
      if ( (unsigned int)sub_C44630(v30 + 24) == 1 )
        goto LABEL_61;
    }
    else
    {
      v32 = *(_QWORD *)(v30 + 24);
      if ( v32 )
      {
        v19 = v32 - 1;
        if ( (v32 & (v32 - 1)) == 0 )
          goto LABEL_61;
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v30 + 8) + 8LL) - 17 > 1 )
      goto LABEL_27;
  }
  else
  {
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v30 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 || *(_BYTE *)v30 > 0x15u )
      goto LABEL_27;
  }
  v33 = sub_AD7630(v30, 1, v19);
  if ( !v33 || *v33 != 17 )
    goto LABEL_27;
  v31 = (__int64)(v33 + 24);
  if ( *((_DWORD *)v33 + 8) > 0x40u )
  {
    v31 = (__int64)(v33 + 24);
    if ( (unsigned int)sub_C44630((__int64)(v33 + 24)) != 1 )
      goto LABEL_27;
  }
  else
  {
    v34 = *((_QWORD *)v33 + 3);
    if ( !v34 || (v34 & (v34 - 1)) != 0 )
      goto LABEL_27;
  }
LABEL_61:
  v21 = *(_DWORD *)(v31 + 8);
  if ( v21 <= 0x40 )
  {
    v69 = *(_QWORD *)v31;
    v104 = *(_DWORD *)(v31 + 8);
    v103 = v69;
  }
  else
  {
    sub_C43990((__int64)&v103, v31);
    v21 = v104;
  }
  v95 = 0;
LABEL_40:
  if ( v14 == 33 )
  {
    v24 = v98;
    v98 = v99;
    v99 = v24;
  }
  v94 = *(_DWORD *)(v99 + 8);
  if ( v94 <= 0x40 )
  {
    v93 = *(_QWORD *)v99 != 0;
    if ( !*(_QWORD *)v99 )
    {
LABEL_45:
      v27 = 1;
      v28 = *(_DWORD *)(v98 + 8);
      goto LABEL_46;
    }
  }
  else
  {
    v86 = v21;
    v25 = sub_C444A0(v99);
    v21 = v86;
    v93 = v94 != v25;
    if ( v94 == v25 )
    {
      v26 = sub_C44630(v99);
      v21 = v86;
      if ( v26 == 1 )
      {
        v94 = *(_DWORD *)(v98 + 8);
LABEL_100:
        if ( v94 <= 0x40 )
        {
          v36 = v94 - 64;
          v29 = *(_QWORD *)v98;
          if ( *(_QWORD *)v98 )
            goto LABEL_71;
          v84 = -1;
          v100 = 0;
          v82 = v94;
LABEL_72:
          if ( v21 > 0x40 )
          {
            v89 = v21;
            v46 = sub_C444A0((__int64)&v103);
            v21 = v89;
            v81 = v46;
            v38 = v89 - v46;
            v83 = v89 - v46 - 1;
          }
          else if ( v103 )
          {
            _BitScanReverse64(&v37, v103);
            LODWORD(v37) = v21 - 64 + (v37 ^ 0x3F);
            v38 = v21 - v37;
            v81 = v37;
            v83 = v21 - v37 - 1;
          }
          else
          {
            v81 = v21;
            v38 = 0;
            v83 = -1;
          }
          v88 = v21;
          v39 = sub_BCB060(v7);
          v40 = sub_BCB060(*(_QWORD *)(v97 + 8));
          v41 = *(_QWORD *)(a2 + 16);
          v21 = v88;
          v42 = 0;
          v43 = v93 + v95 + (v100 != v38) + (v39 != v40);
          if ( v41 )
            v42 = *(_QWORD *)(v41 + 8) == 0;
          v11 = 0;
          if ( v43 > 2 - !v42 )
            goto LABEL_28;
          if ( v95 )
          {
            LOWORD(v111[0]) = 257;
            v49 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v97 + 8), (__int64)&v103);
            v50 = sub_A82350((unsigned int **)a3, (_BYTE *)v97, v49, (__int64)&v107);
            v21 = v88;
            v97 = v50;
          }
          if ( v84 > v83 )
          {
            v96 = v21;
            LOWORD(v111[0]) = 257;
            v51 = sub_A830B0((unsigned int **)a3, v97, v7, (__int64)&v107);
            v106 = 257;
            v52 = sub_AD64C0(*(_QWORD *)(v51 + 8), v100 - v96 + v81, 0);
            v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a3 + 80)
                                                                                               + 32LL))(
                    *(_QWORD *)(a3 + 80),
                    25,
                    v51,
                    v52,
                    0,
                    0);
            if ( !v11 )
            {
              LOWORD(v111[0]) = 257;
              v11 = sub_B504D0(25, v51, v52, (__int64)&v107, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                *(_QWORD *)(a3 + 88),
                v11,
                v105,
                *(_QWORD *)(a3 + 56),
                *(_QWORD *)(a3 + 64));
              v53 = *(unsigned int **)a3;
              v54 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
              if ( *(_QWORD *)a3 != v54 )
              {
                do
                {
                  v55 = *((_QWORD *)v53 + 1);
                  v56 = *v53;
                  v53 += 4;
                  sub_B99FD0(v11, v56, v55);
                }
                while ( (unsigned int *)v54 != v53 );
              }
            }
          }
          else if ( v84 >= v83 )
          {
            LOWORD(v111[0]) = 257;
            v11 = sub_A830B0((unsigned int **)a3, v97, v7, (__int64)&v107);
          }
          else
          {
            v106 = 257;
            v44 = sub_AD64C0(*(_QWORD *)(v97 + 8), v82 - v94 + v38, 0);
            v45 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a3 + 80) + 24LL))(
                    *(_QWORD *)(a3 + 80),
                    26,
                    v97,
                    v44,
                    0);
            if ( !v45 )
            {
              LOWORD(v111[0]) = 257;
              v74 = sub_B504D0(26, v97, v44, (__int64)&v107, 0, 0);
              v45 = sub_1157250((__int64 *)a3, v74, (__int64)v105);
            }
            LOWORD(v111[0]) = 257;
            v11 = sub_A830B0((unsigned int **)a3, v45, v7, (__int64)&v107);
          }
          if ( v93 )
          {
            v106 = 257;
            v57 = sub_AD8D80(*(_QWORD *)(v11 + 8), v98);
            v58 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a3 + 80) + 16LL))(
                    *(_QWORD *)(a3 + 80),
                    30,
                    v11,
                    v57);
            if ( !v58 )
            {
              LOWORD(v111[0]) = 257;
              v58 = sub_B504D0(30, v11, v57, (__int64)&v107, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
                *(_QWORD *)(a3 + 88),
                v58,
                v105,
                *(_QWORD *)(a3 + 56),
                *(_QWORD *)(a3 + 64));
              v70 = *(unsigned int **)a3;
              v71 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
              if ( *(_QWORD *)a3 != v71 )
              {
                do
                {
                  v72 = *((_QWORD *)v70 + 1);
                  v73 = *v70;
                  v70 += 4;
                  sub_B99FD0(v58, v73, v72);
                }
                while ( (unsigned int *)v71 != v70 );
              }
            }
            v21 = v104;
            v11 = v58;
            goto LABEL_28;
          }
          goto LABEL_86;
        }
LABEL_104:
        v91 = v21;
        v48 = sub_C444A0(v98);
        v21 = v91;
        v82 = v48;
        v100 = v94 - v48;
        v84 = v94 - v48 - 1;
        goto LABEL_72;
      }
      goto LABEL_45;
    }
  }
  v28 = *(_DWORD *)(v98 + 8);
  if ( v28 <= 0x40 )
  {
    if ( !*(_QWORD *)v98 )
      goto LABEL_67;
LABEL_89:
    v11 = 0;
    if ( v21 != v94 )
      goto LABEL_28;
    if ( !v95 )
      goto LABEL_117;
    v11 = *(_QWORD *)(a2 + 16);
    if ( !v11 )
      goto LABEL_28;
    if ( !*(_QWORD *)(v11 + 8) )
    {
LABEL_117:
      v59 = sub_AD8D80(v7, v99);
      v60 = sub_AD8D80(v7, v98);
      v61 = v7;
      v62 = 29;
      v101 = v60;
      v63 = (_BYTE *)sub_AD8D80(v61, (__int64)&v103);
      v64 = a1;
      v65 = (unsigned int *)&unk_3F92D30;
      v66 = v64;
      while ( 1 )
      {
        v67 = sub_B43CC0(v66);
        if ( v101 == sub_96E6C0(v62, v59, v63, v67) )
          break;
        if ( &unk_3F92D40 == (_UNKNOWN *)++v65 )
          goto LABEL_27;
        v62 = *v65;
      }
      if ( v95 )
      {
        LOWORD(v111[0]) = 257;
        v97 = sub_A82350((unsigned int **)a3, (_BYTE *)v97, v63, (__int64)&v107);
      }
      v106 = 257;
      v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a3 + 80) + 16LL))(
              *(_QWORD *)(a3 + 80),
              v62,
              v59,
              v97);
      if ( !v11 )
      {
        LOWORD(v111[0]) = 257;
        v11 = sub_B504D0(v62, v59, v97, (__int64)&v107, 0, 0);
        if ( (unsigned __int8)sub_920620(v11) )
        {
          v75 = *(_QWORD *)(a3 + 96);
          v76 = *(_DWORD *)(a3 + 104);
          if ( v75 )
            sub_B99FD0(v11, 3u, v75);
          sub_B45150(v11, v76);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v11,
          v105,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v77 = *(unsigned int **)a3;
        v78 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        if ( *(_QWORD *)a3 != v78 )
        {
          do
          {
            v79 = *((_QWORD *)v77 + 1);
            v80 = *v77;
            v77 += 4;
            sub_B99FD0(v11, v80, v79);
          }
          while ( (unsigned int *)v78 != v77 );
        }
      }
LABEL_86:
      v21 = v104;
      goto LABEL_28;
    }
    goto LABEL_93;
  }
  v87 = v21;
  v35 = sub_C444A0(v98);
  v21 = v87;
  if ( v28 != v35 )
    goto LABEL_89;
LABEL_67:
  if ( v94 > 0x40 )
  {
    v92 = v21;
    v68 = sub_C44630(v99);
    v21 = v92;
    if ( v68 == 1 )
    {
      v98 = v99;
      goto LABEL_104;
    }
  }
  else
  {
    v29 = *(_QWORD *)v99;
    if ( *(_QWORD *)v99 && (v29 & (v29 - 1)) == 0 )
      goto LABEL_70;
  }
  v27 = 0;
LABEL_46:
  if ( v28 > 0x40 )
  {
    v85 = v27;
    v90 = v21;
    v47 = sub_C44630(v98);
    v21 = v90;
    if ( v47 == 1 )
    {
      if ( v85 )
      {
        v94 = v28;
        goto LABEL_104;
      }
      goto LABEL_99;
    }
LABEL_93:
    v11 = 0;
    goto LABEL_28;
  }
  v11 = 0;
  v29 = *(_QWORD *)v98;
  if ( *(_QWORD *)v98 && (v29 & (v29 - 1)) == 0 )
  {
    if ( v27 )
    {
      v94 = v28;
      v99 = v98;
LABEL_70:
      v98 = v99;
      v36 = v94 - 64;
LABEL_71:
      _BitScanReverse64(&v29, v29);
      v82 = (v29 ^ 0x3F) + v36;
      v100 = v94 - v82;
      v84 = v94 - v82 - 1;
      goto LABEL_72;
    }
LABEL_99:
    v98 = v99;
    goto LABEL_100;
  }
LABEL_28:
  if ( v21 > 0x40 && v103 )
    j_j___libc_free_0_0(v103);
  return v11;
}
