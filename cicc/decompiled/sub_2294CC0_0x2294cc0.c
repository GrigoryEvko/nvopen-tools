// Function: sub_2294CC0
// Address: 0x2294cc0
//
__int64 __fastcall sub_2294CC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        char *a7,
        __int64 a8)
{
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r13
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned int v15; // ebx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  unsigned int v24; // edx
  unsigned __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 v27; // r12
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  unsigned int v30; // eax
  unsigned __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // r12
  __int64 v34; // r13
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 v37; // r12
  __int64 v38; // rbx
  unsigned __int64 v39; // r12
  __int64 v40; // rbx
  unsigned __int64 v41; // r12
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // rax
  unsigned int v48; // eax
  unsigned int v49; // eax
  unsigned int *v50; // rdi
  unsigned __int64 v51; // rax
  unsigned int v52; // eax
  unsigned int v53; // eax
  unsigned int *v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rdx
  unsigned int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned int v61; // [rsp+20h] [rbp-1F0h]
  char v62; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 v64; // [rsp+50h] [rbp-1C0h]
  char v65; // [rsp+5Dh] [rbp-1B3h]
  char v66; // [rsp+5Eh] [rbp-1B2h]
  unsigned __int8 v67; // [rsp+5Fh] [rbp-1B1h]
  unsigned __int64 v68; // [rsp+60h] [rbp-1B0h] BYREF
  unsigned int v69; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v70; // [rsp+70h] [rbp-1A0h] BYREF
  unsigned int v71; // [rsp+78h] [rbp-198h]
  unsigned __int64 v72; // [rsp+80h] [rbp-190h] BYREF
  unsigned int v73; // [rsp+88h] [rbp-188h]
  unsigned __int64 v74; // [rsp+90h] [rbp-180h] BYREF
  unsigned int v75; // [rsp+98h] [rbp-178h]
  unsigned __int64 v76; // [rsp+A0h] [rbp-170h] BYREF
  unsigned int v77; // [rsp+A8h] [rbp-168h]
  unsigned __int64 v78; // [rsp+B0h] [rbp-160h] BYREF
  unsigned int v79; // [rsp+B8h] [rbp-158h]
  unsigned __int64 v80; // [rsp+C0h] [rbp-150h] BYREF
  unsigned int v81; // [rsp+C8h] [rbp-148h]
  __int64 v82; // [rsp+D0h] [rbp-140h] BYREF
  unsigned int v83; // [rsp+D8h] [rbp-138h]
  unsigned __int64 v84; // [rsp+E0h] [rbp-130h] BYREF
  unsigned int v85; // [rsp+E8h] [rbp-128h]
  __int64 v86; // [rsp+F0h] [rbp-120h] BYREF
  unsigned int v87; // [rsp+F8h] [rbp-118h]
  __int64 v88[2]; // [rsp+100h] [rbp-110h] BYREF
  __int64 v89[2]; // [rsp+110h] [rbp-100h] BYREF
  __int64 v90[2]; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v91[2]; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v92[2]; // [rsp+140h] [rbp-D0h] BYREF
  unsigned __int64 v93; // [rsp+150h] [rbp-C0h] BYREF
  unsigned int v94; // [rsp+158h] [rbp-B8h]
  unsigned __int64 v95; // [rsp+160h] [rbp-B0h] BYREF
  unsigned int v96; // [rsp+168h] [rbp-A8h]
  __int64 v97[2]; // [rsp+170h] [rbp-A0h] BYREF
  _BYTE *v98; // [rsp+180h] [rbp-90h] BYREF
  __int64 v99; // [rsp+188h] [rbp-88h]
  _BYTE v100[32]; // [rsp+190h] [rbp-80h] BYREF
  _BYTE *v101; // [rsp+1B0h] [rbp-60h] BYREF
  __int64 v102; // [rsp+1B8h] [rbp-58h]
  _BYTE v103[80]; // [rsp+1C0h] [rbp-50h] BYREF

  *(_BYTE *)(a8 + 43) = 0;
  v9 = sub_DCC810(*(__int64 **)(a1 + 8), a5, a4, 0, 0);
  v67 = 0;
  if ( *((_WORD *)v9 + 12) || *(_WORD *)(a2 + 24) || *(_WORD *)(a3 + 24) )
    return v67;
  v10 = *(_QWORD *)(a2 + 32);
  v11 = (__int64)v9;
  v69 = 1;
  v68 = 0;
  v71 = 1;
  v12 = *(_DWORD *)(v10 + 32);
  v70 = 0;
  v73 = 1;
  v72 = 0;
  v75 = v12;
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v74, (const void **)(v10 + 24));
  else
    v74 = *(_QWORD *)(v10 + 24);
  v13 = *(_QWORD *)(a3 + 32);
  v77 = *(_DWORD *)(v13 + 32);
  if ( v77 > 0x40 )
    sub_C43780((__int64)&v76, (const void **)(v13 + 24));
  else
    v76 = *(_QWORD *)(v13 + 24);
  v14 = *(_QWORD *)(v11 + 32);
  v79 = *(_DWORD *)(v14 + 32);
  if ( v79 > 0x40 )
    sub_C43780((__int64)&v78, (const void **)(v14 + 24));
  else
    v78 = *(_QWORD *)(v14 + 24);
  v15 = v75;
  v67 = sub_228B3E0(v75, (__int64 *)&v74, (__int64)&v76, (__int64)&v78, (__int64)&v68, (__int64)&v70, (__int64)&v72);
  if ( !v67 )
  {
    v81 = v15;
    v61 = v15 - 1;
    v62 = (v15 - 1) & 0x3F;
    if ( v15 > 0x40 )
    {
      sub_C43690((__int64)&v80, 1, 1);
      v20 = sub_D95540(v11);
      v18 = sub_228E3C0(a1, a6, v20);
      if ( !v18 )
      {
        v83 = v15;
        v66 = 0;
        goto LABEL_18;
      }
    }
    else
    {
      v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & 1;
      if ( !v15 )
        v16 = 0;
      v80 = v16;
      v17 = sub_D95540(v11);
      v18 = sub_228E3C0(a1, a6, v17);
      v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
      if ( !v18 )
      {
        v83 = v15;
        v66 = 0;
        goto LABEL_98;
      }
    }
    v43 = v18[4];
    if ( v81 <= 0x40 && (v44 = *(_DWORD *)(v43 + 32), v44 <= 0x40) )
    {
      v59 = *(_QWORD *)(v43 + 24);
      v81 = v44;
      v80 = v59;
    }
    else
    {
      sub_C43990((__int64)&v80, v43 + 24);
    }
    v83 = v15;
    if ( v15 <= 0x40 )
    {
      v66 = 1;
      v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
LABEL_98:
      v64 = v19;
      v45 = v19 & 1;
      if ( !v15 )
        v45 = 0;
      v82 = v45;
      v46 = sub_D95540(v11);
      v22 = sub_228E3C0(a1, a7, v46);
      v25 = v64;
      if ( !v22 )
      {
        v85 = v15;
        v65 = 0;
        goto LABEL_24;
      }
      goto LABEL_19;
    }
    v66 = 1;
LABEL_18:
    sub_C43690((__int64)&v82, 1, 1);
    v21 = sub_D95540(v11);
    v22 = sub_228E3C0(a1, a7, v21);
    if ( !v22 )
    {
      v85 = v15;
      v65 = 0;
      goto LABEL_129;
    }
LABEL_19:
    v23 = v22[4];
    if ( v83 <= 0x40 && (v24 = *(_DWORD *)(v23 + 32), v24 <= 0x40) )
    {
      v60 = *(_QWORD *)(v23 + 24);
      v83 = v24;
      v82 = v60;
    }
    else
    {
      sub_C43990((__int64)&v82, v23 + 24);
    }
    v85 = v15;
    if ( v15 <= 0x40 )
    {
      v65 = 1;
      v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
LABEL_24:
      v26 = v15;
      if ( !v15 )
        v25 = 0;
      v84 = v25;
LABEL_27:
      v27 = 1LL << v62;
      v28 = ~(1LL << v62);
      if ( v26 > 0x40 )
        *(_QWORD *)(v84 + 8LL * (v61 >> 6)) &= v28;
      else
        v84 &= v28;
      v87 = v15;
      if ( v15 > 0x40 )
      {
        sub_C43690((__int64)&v86, 0, 0);
        if ( v87 > 0x40 )
        {
          *(_QWORD *)(v86 + 8LL * (v61 >> 6)) |= v27;
LABEL_32:
          sub_C4A3E0((__int64)v88, (__int64)&v78, (__int64)&v68);
          sub_C472A0((__int64)v89, (__int64)&v70, v88);
          sub_C472A0((__int64)v90, (__int64)&v72, v88);
          v98 = v100;
          v99 = 0x200000000LL;
          v101 = v103;
          v102 = 0x200000000LL;
          sub_C4A3E0((__int64)v91, (__int64)&v76, (__int64)&v68);
          if ( sub_AAD930((__int64)v91, 0) )
          {
            sub_9865C0((__int64)&v93, (__int64)v89);
            if ( v94 > 0x40 )
            {
              sub_C43D10((__int64)&v93);
            }
            else
            {
              v29 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v94) & ~v93;
              if ( !v94 )
                v29 = 0;
              v93 = v29;
            }
            sub_C46250((__int64)&v93);
            v30 = v94;
            v94 = 0;
            v96 = v30;
            v95 = v93;
            sub_228AEF0((__int64)v97, (__int64)&v95, (__int64)v91);
            sub_2293630((unsigned int *)&v98, (unsigned __int64)v97);
            sub_969240(v97);
            sub_969240((__int64 *)&v95);
            sub_969240((__int64 *)&v93);
            if ( !v66 )
              goto LABEL_38;
            sub_9865C0((__int64)&v93, (__int64)&v80);
            sub_C46B40((__int64)&v93, v89);
            v57 = v94;
            v94 = 0;
            v96 = v57;
            v95 = v93;
            sub_228B180((__int64)v97, (__int64)&v95, (__int64)v91);
            v54 = (unsigned int *)&v101;
          }
          else
          {
            sub_9865C0((__int64)&v93, (__int64)v89);
            if ( v94 > 0x40 )
            {
              sub_C43D10((__int64)&v93);
            }
            else
            {
              v51 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v94) & ~v93;
              if ( !v94 )
                v51 = 0;
              v93 = v51;
            }
            sub_C46250((__int64)&v93);
            v52 = v94;
            v94 = 0;
            v96 = v52;
            v95 = v93;
            sub_228B180((__int64)v97, (__int64)&v95, (__int64)v91);
            sub_2293630((unsigned int *)&v101, (unsigned __int64)v97);
            sub_969240(v97);
            sub_969240((__int64 *)&v95);
            sub_969240((__int64 *)&v93);
            if ( !v66 )
            {
LABEL_38:
              sub_C4A3E0((__int64)v92, (__int64)&v74, (__int64)&v68);
              if ( sub_AAD930((__int64)v92, 0) )
              {
                sub_9865C0((__int64)&v93, (__int64)v90);
                if ( v94 > 0x40 )
                {
                  sub_C43D10((__int64)&v93);
                }
                else
                {
                  v31 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v94) & ~v93;
                  if ( !v94 )
                    v31 = 0;
                  v93 = v31;
                }
                sub_C46250((__int64)&v93);
                v32 = v94;
                v94 = 0;
                v96 = v32;
                v95 = v93;
                sub_228AEF0((__int64)v97, (__int64)&v95, (__int64)v92);
                sub_2293630((unsigned int *)&v98, (unsigned __int64)v97);
                sub_969240(v97);
                sub_969240((__int64 *)&v95);
                sub_969240((__int64 *)&v93);
                if ( !v65 )
                  goto LABEL_44;
                sub_9865C0((__int64)&v93, (__int64)&v82);
                sub_C46B40((__int64)&v93, v90);
                v58 = v94;
                v94 = 0;
                v96 = v58;
                v95 = v93;
                sub_228B180((__int64)v97, (__int64)&v95, (__int64)v92);
                v50 = (unsigned int *)&v101;
              }
              else
              {
                sub_9865C0((__int64)&v93, (__int64)v90);
                if ( v94 > 0x40 )
                {
                  sub_C43D10((__int64)&v93);
                }
                else
                {
                  v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v94) & ~v93;
                  if ( !v94 )
                    v47 = 0;
                  v93 = v47;
                }
                sub_C46250((__int64)&v93);
                v48 = v94;
                v94 = 0;
                v96 = v48;
                v95 = v93;
                sub_228B180((__int64)v97, (__int64)&v95, (__int64)v92);
                sub_2293630((unsigned int *)&v101, (unsigned __int64)v97);
                sub_969240(v97);
                sub_969240((__int64 *)&v95);
                sub_969240((__int64 *)&v93);
                if ( !v65 )
                  goto LABEL_44;
                sub_9865C0((__int64)&v93, (__int64)&v82);
                sub_C46B40((__int64)&v93, v90);
                v49 = v94;
                v94 = 0;
                v96 = v49;
                v95 = v93;
                sub_228AEF0((__int64)v97, (__int64)&v95, (__int64)v92);
                v50 = (unsigned int *)&v98;
              }
              sub_2293630(v50, (unsigned __int64)v97);
              sub_969240(v97);
              sub_969240((__int64 *)&v95);
              sub_969240((__int64 *)&v93);
LABEL_44:
              if ( (_DWORD)v99 )
              {
                v33 = (unsigned int)v102;
                if ( (_DWORD)v102 )
                {
                  v34 = (__int64)v98;
                  v35 = (__int64)&v98[16 * (unsigned int)v99 - 16];
                  if ( (int)sub_C4C880((__int64)v98, v35) <= 0 )
                    v34 = v35;
                  if ( v87 <= 0x40 && *(_DWORD *)(v34 + 8) <= 0x40u )
                  {
                    v56 = *(_QWORD *)v34;
                    v87 = *(_DWORD *)(v34 + 8);
                    v86 = v56;
                  }
                  else
                  {
                    sub_C43990((__int64)&v86, v34);
                    v33 = (unsigned int)v102;
                  }
                  v36 = (__int64)v101;
                  v37 = (__int64)&v101[16 * v33 - 16];
                  if ( (int)sub_C4C880((__int64)v101, v37) >= 0 )
                    v36 = v37;
                  if ( v85 <= 0x40 && *(_DWORD *)(v36 + 8) <= 0x40u )
                  {
                    v55 = *(_QWORD *)v36;
                    v85 = *(_DWORD *)(v36 + 8);
                    v84 = v55;
                  }
                  else
                  {
                    sub_C43990((__int64)&v84, v36);
                  }
                  v67 = (int)sub_C4C880((__int64)&v86, (__int64)&v84) > 0;
                }
              }
              sub_969240(v92);
              sub_969240(v91);
              v38 = (__int64)v101;
              v39 = (unsigned __int64)&v101[16 * (unsigned int)v102];
              if ( v101 != (_BYTE *)v39 )
              {
                do
                {
                  v39 -= 16LL;
                  if ( *(_DWORD *)(v39 + 8) > 0x40u && *(_QWORD *)v39 )
                    j_j___libc_free_0_0(*(_QWORD *)v39);
                }
                while ( v38 != v39 );
                v39 = (unsigned __int64)v101;
              }
              if ( (_BYTE *)v39 != v103 )
                _libc_free(v39);
              v40 = (__int64)v98;
              v41 = (unsigned __int64)&v98[16 * (unsigned int)v99];
              if ( v98 != (_BYTE *)v41 )
              {
                do
                {
                  v41 -= 16LL;
                  if ( *(_DWORD *)(v41 + 8) > 0x40u && *(_QWORD *)v41 )
                    j_j___libc_free_0_0(*(_QWORD *)v41);
                }
                while ( v40 != v41 );
                v41 = (unsigned __int64)v98;
              }
              if ( (_BYTE *)v41 != v100 )
                _libc_free(v41);
              sub_969240(v90);
              sub_969240(v89);
              sub_969240(v88);
              sub_969240(&v86);
              sub_969240((__int64 *)&v84);
              sub_969240(&v82);
              sub_969240((__int64 *)&v80);
              goto LABEL_74;
            }
            sub_9865C0((__int64)&v93, (__int64)&v80);
            sub_C46B40((__int64)&v93, v89);
            v53 = v94;
            v94 = 0;
            v96 = v53;
            v95 = v93;
            sub_228AEF0((__int64)v97, (__int64)&v95, (__int64)v91);
            v54 = (unsigned int *)&v98;
          }
          sub_2293630(v54, (unsigned __int64)v97);
          sub_969240(v97);
          sub_969240((__int64 *)&v95);
          sub_969240((__int64 *)&v93);
          goto LABEL_38;
        }
      }
      else
      {
        v86 = 0;
      }
      v86 |= v27;
      goto LABEL_32;
    }
    v65 = 1;
LABEL_129:
    sub_C43690((__int64)&v84, -1, 1);
    v26 = v85;
    goto LABEL_27;
  }
LABEL_74:
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  return v67;
}
