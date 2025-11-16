// Function: sub_1197740
// Address: 0x1197740
//
__int64 __fastcall sub_1197740(__int64 a1, unsigned int a2, unsigned __int8 a3, __int64 a4)
{
  unsigned int v5; // r13d
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v13; // r15
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdx
  unsigned int v17; // esi
  unsigned __int8 v18; // r14
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 *v29; // rax
  __int64 v30; // rbx
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rdi
  unsigned int v34; // r9d
  _QWORD *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int v41; // r14d
  __int64 v42; // r15
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rcx
  __int64 *v47; // rdi
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 *v53; // rsi
  unsigned int v54; // eax
  unsigned int v55; // r8d
  unsigned __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // r12
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  __int64 v63; // rdi
  unsigned __int64 v64; // rdi
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rcx
  _BYTE *v75; // rax
  _BYTE *v76; // rsi
  unsigned int v77; // eax
  unsigned int v78; // r14d
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // r15
  unsigned int v83; // r14d
  __int64 v84; // rbx
  __int64 v85; // r14
  __int64 v86; // rdx
  unsigned int v87; // esi
  unsigned int v88; // [rsp+4h] [rbp-BCh]
  unsigned int v89; // [rsp+4h] [rbp-BCh]
  __int64 *v90; // [rsp+8h] [rbp-B8h]
  char v91; // [rsp+8h] [rbp-B8h]
  __int64 v92; // [rsp+10h] [rbp-B0h]
  __int64 v93; // [rsp+10h] [rbp-B0h]
  __int64 v94; // [rsp+18h] [rbp-A8h]
  _QWORD *v95; // [rsp+18h] [rbp-A8h]
  char v96; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v97; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v98; // [rsp+28h] [rbp-98h]
  unsigned __int64 v99; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v100; // [rsp+38h] [rbp-88h]
  __int16 v101; // [rsp+50h] [rbp-70h]
  __int64 v102[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v103; // [rsp+80h] [rbp-40h]

  v5 = a2;
  if ( *(_BYTE *)a1 > 0x15u )
  {
    sub_F15FC0(*(_QWORD *)(a4 + 40), a1);
    v18 = *(_BYTE *)a1;
    switch ( *(_BYTE *)a1 )
    {
      case '.':
        v103 = 257;
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v47 = *(__int64 **)(a1 - 8);
        else
          v47 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
        v48 = a1 + 24;
        v49 = sub_B50550(*v47, (__int64)v102, 0, 0);
        v50 = *(_QWORD *)(a1 + 48);
        v95 = (_QWORD *)v49;
        v102[0] = v50;
        if ( v50 )
        {
          sub_B96E90((__int64)v102, v50, 1);
          v51 = v95[6];
          v52 = (__int64)(v95 + 6);
          if ( !v51 )
            goto LABEL_63;
        }
        else
        {
          v51 = *(_QWORD *)(v49 + 48);
          v52 = v49 + 48;
          if ( !v51 )
            goto LABEL_65;
        }
        v93 = v52;
        sub_B91220(v52, v51);
        v52 = v93;
LABEL_63:
        v53 = (unsigned __int8 *)v102[0];
        v95[6] = v102[0];
        if ( v53 )
          sub_B976B0((__int64)v102, v53, v52);
LABEL_65:
        sub_B44220(v95, a1 + 24, 0);
        v102[0] = (__int64)v95;
        sub_1196C30(*(_QWORD *)(a4 + 40) + 2096LL, v102);
        v54 = sub_BCB060(*(_QWORD *)(a1 + 8));
        v100 = v54;
        v55 = v54 - v5;
        if ( v54 > 0x40 )
        {
          v89 = v54 - v5;
          v91 = v54;
          sub_C43690((__int64)&v99, 0, 0);
          v55 = v89;
          LOBYTE(v54) = v91;
        }
        else
        {
          v99 = 0;
        }
        if ( v55 )
        {
          if ( v55 > 0x40 )
          {
            sub_C43C90(&v99, 0, v55);
          }
          else
          {
            v56 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 + 64 - (unsigned __int8)v54);
            if ( v100 > 0x40 )
              *(_QWORD *)v99 |= v56;
            else
              v99 |= v56;
          }
        }
        v57 = *(_QWORD *)(a1 + 8);
        v103 = 257;
        v58 = sub_AD8D80(v57, (__int64)&v99);
        v11 = sub_B504D0(28, (__int64)v95, v58, (__int64)v102, 0, 0);
        sub_BD6B90((unsigned __int8 *)v11, (unsigned __int8 *)a1);
        v59 = *(_QWORD *)(a1 + 48);
        v102[0] = v59;
        if ( v59 )
        {
          v60 = v11 + 48;
          sub_B96E90((__int64)v102, v59, 1);
          v61 = *(_QWORD *)(v11 + 48);
          if ( !v61 )
            goto LABEL_74;
        }
        else
        {
          v61 = *(_QWORD *)(v11 + 48);
          v60 = v11 + 48;
          if ( !v61 )
            goto LABEL_76;
        }
        sub_B91220(v60, v61);
LABEL_74:
        v62 = (unsigned __int8 *)v102[0];
        *(_QWORD *)(v11 + 48) = v102[0];
        if ( v62 )
          sub_B976B0((__int64)v102, v62, v60);
LABEL_76:
        sub_B44220((_QWORD *)v11, v48, 0);
        v63 = *(_QWORD *)(a4 + 40);
        v102[0] = v11;
        sub_1196C30(v63 + 2096, v102);
        if ( v100 > 0x40 )
        {
          v64 = v99;
          if ( v99 )
            goto LABEL_78;
        }
        return v11;
      case '6':
      case '7':
        v29 = *(__int64 **)(a4 + 32);
        v30 = *(_QWORD *)(a1 + 8);
        v90 = v29;
        v31 = sub_BCB060(v30);
        v33 = *(_QWORD *)(a1 - 32);
        v34 = v31;
        if ( *(_BYTE *)v33 == 17 )
        {
          v92 = v33 + 24;
        }
        else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v33 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v33 <= 0x15u )
        {
          v88 = v31;
          v75 = sub_AD7630(v33, 0, v32);
          v34 = v88;
          if ( v75 )
          {
            v76 = v75 + 24;
            if ( *v75 != 17 )
              v76 = 0;
            v92 = (__int64)v76;
          }
        }
        v35 = *(_QWORD **)v92;
        if ( *(_DWORD *)(v92 + 8) > 0x40u )
          v35 = (_QWORD *)*v35;
        v36 = (unsigned int)v35;
        if ( a3 == (v18 == 54) )
        {
          v77 = v5 + (_DWORD)v35;
          if ( v34 <= v77 )
            return sub_AD6530(v30, v36);
          v37 = v77;
        }
        else
        {
          v37 = (unsigned int)v35 - v5;
          if ( v5 == (_DWORD)v35 )
          {
            if ( v18 == 54 )
            {
              v98 = v34;
              v78 = v34 - v5;
              if ( v34 > 0x40 )
              {
                v96 = v34;
                sub_C43690((__int64)&v97, 0, 0);
                LOBYTE(v34) = v96;
              }
              else
              {
                v97 = 0;
              }
              if ( v78 )
              {
                if ( v78 > 0x40 )
                {
                  sub_C43C90(&v97, 0, v78);
                }
                else
                {
                  v79 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 + 64 - (unsigned __int8)v34);
                  if ( v98 > 0x40 )
                    *(_QWORD *)v97 |= v79;
                  else
                    v97 |= v79;
                }
              }
            }
            else
            {
              v98 = v34;
              v83 = v5 - v34;
              if ( v34 > 0x40 )
              {
                sub_C43690((__int64)&v97, 0, 0);
                v34 = v98;
                v5 = v83 + v98;
              }
              else
              {
                v97 = 0;
              }
              if ( v5 != v34 )
              {
                if ( v5 > 0x3F || v34 > 0x40 )
                  sub_C43C90(&v97, v5, v34);
                else
                  v97 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v83 + 64) << v5;
              }
            }
            v101 = 257;
            v80 = sub_AD8D80(v30, (__int64)&v97);
            v81 = *(_QWORD *)(a1 - 64);
            v82 = v80;
            v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v90[10] + 16LL))(
                    v90[10],
                    28,
                    v81,
                    v80);
            if ( !v11 )
            {
              v103 = 257;
              v11 = sub_B504D0(28, v81, v82, (__int64)v102, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v90[11] + 16LL))(
                v90[11],
                v11,
                &v99,
                v90[7],
                v90[8]);
              v84 = *v90;
              v85 = *v90 + 16LL * *((unsigned int *)v90 + 2);
              if ( *v90 != v85 )
              {
                do
                {
                  v86 = *(_QWORD *)(v84 + 8);
                  v87 = *(_DWORD *)v84;
                  v84 += 16;
                  sub_B99FD0(v11, v87, v86);
                }
                while ( v85 != v84 );
              }
            }
            if ( *(_BYTE *)v11 > 0x1Cu )
            {
              sub_B444E0((_QWORD *)v11, a1 + 24, 0);
              sub_BD6B90((unsigned __int8 *)v11, (unsigned __int8 *)a1);
            }
            if ( v98 > 0x40 )
            {
              v64 = v97;
              if ( v97 )
LABEL_78:
                j_j___libc_free_0_0(v64);
            }
            return v11;
          }
        }
        v38 = sub_AD64C0(v30, v37, 0);
        if ( *(_QWORD *)(a1 - 32) )
        {
          v39 = *(_QWORD *)(a1 - 24);
          **(_QWORD **)(a1 - 16) = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 16) = *(_QWORD *)(a1 - 16);
        }
        *(_QWORD *)(a1 - 32) = v38;
        if ( v38 )
        {
          v40 = *(_QWORD *)(v38 + 16);
          *(_QWORD *)(a1 - 24) = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = a1 - 24;
          *(_QWORD *)(a1 - 16) = v38 + 16;
          *(_QWORD *)(v38 + 16) = a1 - 32;
        }
        if ( v18 == 54 )
        {
          sub_B447F0((unsigned __int8 *)a1, 0);
          v11 = a1;
          sub_B44850((unsigned __int8 *)a1, 0);
        }
        else
        {
          sub_B448B0(a1, 0);
          return a1;
        }
        return v11;
      case '9':
      case ':':
      case ';':
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v19 = *(_QWORD **)(a1 - 8);
        else
          v19 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
        v20 = sub_1197740(*v19, a2, a3, a4);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v21 = *(_QWORD *)(a1 - 8);
        else
          v21 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v21 )
        {
          v22 = *(_QWORD *)(v21 + 8);
          **(_QWORD **)(v21 + 16) = v22;
          if ( v22 )
            *(_QWORD *)(v22 + 16) = *(_QWORD *)(v21 + 16);
        }
        *(_QWORD *)v21 = v20;
        if ( v20 )
        {
          v23 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(v21 + 8) = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = v21 + 8;
          *(_QWORD *)(v21 + 16) = v20 + 16;
          *(_QWORD *)(v20 + 16) = v21;
        }
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v24 = *(_QWORD *)(a1 - 8);
        else
          v24 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v25 = sub_1197740(*(_QWORD *)(v24 + 32), a2, a3, a4);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v26 = *(_QWORD *)(a1 - 8);
        else
          v26 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)(v26 + 32) )
        {
          v27 = *(_QWORD *)(v26 + 40);
          **(_QWORD **)(v26 + 48) = v27;
          if ( v27 )
            *(_QWORD *)(v27 + 16) = *(_QWORD *)(v26 + 48);
        }
        *(_QWORD *)(v26 + 32) = v25;
        if ( !v25 )
          return a1;
        v28 = *(_QWORD *)(v25 + 16);
        *(_QWORD *)(v26 + 40) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = v26 + 40;
        *(_QWORD *)(v26 + 48) = v25 + 16;
        v11 = a1;
        *(_QWORD *)(v25 + 16) = v26 + 32;
        return v11;
      case 'T':
        if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
        {
          v41 = a3;
          v42 = 0;
          v94 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
          do
          {
            v43 = sub_1197740(*(_QWORD *)(*(_QWORD *)(a1 - 8) + v42), a2, v41, a4);
            v44 = v42 + *(_QWORD *)(a1 - 8);
            if ( *(_QWORD *)v44 )
            {
              v45 = *(_QWORD *)(v44 + 8);
              **(_QWORD **)(v44 + 16) = v45;
              if ( v45 )
                *(_QWORD *)(v45 + 16) = *(_QWORD *)(v44 + 16);
            }
            *(_QWORD *)v44 = v43;
            if ( v43 )
            {
              v46 = *(_QWORD *)(v43 + 16);
              *(_QWORD *)(v44 + 8) = v46;
              if ( v46 )
                *(_QWORD *)(v46 + 16) = v44 + 8;
              *(_QWORD *)(v44 + 16) = v43 + 16;
              *(_QWORD *)(v43 + 16) = v44;
            }
            v42 += 32;
          }
          while ( v94 != v42 );
        }
        return a1;
      case 'V':
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v65 = *(_QWORD *)(a1 - 8);
        else
          v65 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v66 = sub_1197740(*(_QWORD *)(v65 + 32), a2, a3, a4);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v67 = *(_QWORD *)(a1 - 8);
        else
          v67 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)(v67 + 32) )
        {
          v68 = *(_QWORD *)(v67 + 40);
          **(_QWORD **)(v67 + 48) = v68;
          if ( v68 )
            *(_QWORD *)(v68 + 16) = *(_QWORD *)(v67 + 48);
        }
        *(_QWORD *)(v67 + 32) = v66;
        if ( v66 )
        {
          v69 = *(_QWORD *)(v66 + 16);
          *(_QWORD *)(v67 + 40) = v69;
          if ( v69 )
            *(_QWORD *)(v69 + 16) = v67 + 40;
          *(_QWORD *)(v67 + 48) = v66 + 16;
          *(_QWORD *)(v66 + 16) = v67 + 32;
        }
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v70 = *(_QWORD *)(a1 - 8);
        else
          v70 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v71 = sub_1197740(*(_QWORD *)(v70 + 64), a2, a3, a4);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v72 = *(_QWORD *)(a1 - 8);
        else
          v72 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)(v72 + 64) )
        {
          v73 = *(_QWORD *)(v72 + 72);
          **(_QWORD **)(v72 + 80) = v73;
          if ( v73 )
            *(_QWORD *)(v73 + 16) = *(_QWORD *)(v72 + 80);
        }
        *(_QWORD *)(v72 + 64) = v71;
        if ( v71 )
        {
          v74 = *(_QWORD *)(v71 + 16);
          *(_QWORD *)(v72 + 72) = v74;
          if ( v74 )
            *(_QWORD *)(v74 + 16) = v72 + 72;
          *(_QWORD *)(v72 + 80) = v71 + 16;
          *(_QWORD *)(v71 + 16) = v72 + 64;
        }
        return a1;
      default:
        BUG();
    }
  }
  v8 = *(_QWORD *)(a4 + 32);
  v9 = *(_QWORD *)(a1 + 8);
  if ( a3 )
  {
    v101 = 257;
    v13 = sub_AD64C0(v9, a2, 0);
    v11 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v8 + 80) + 32LL))(
            *(_QWORD *)(v8 + 80),
            25,
            a1,
            v13,
            0,
            0);
    if ( !v11 )
    {
      v103 = 257;
      v11 = sub_B504D0(25, a1, v13, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 88) + 16LL))(
        *(_QWORD *)(v8 + 88),
        v11,
        &v99,
        *(_QWORD *)(v8 + 56),
        *(_QWORD *)(v8 + 64));
      v14 = *(_QWORD *)v8;
      v15 = *(_QWORD *)v8 + 16LL * *(unsigned int *)(v8 + 8);
      if ( *(_QWORD *)v8 != v15 )
      {
        do
        {
          v16 = *(_QWORD *)(v14 + 8);
          v17 = *(_DWORD *)v14;
          v14 += 16;
          sub_B99FD0(v11, v17, v16);
        }
        while ( v15 != v14 );
      }
    }
  }
  else
  {
    v103 = 257;
    v10 = sub_AD64C0(v9, a2, 0);
    return sub_F94560((__int64 *)v8, a1, v10, (__int64)v102, 0);
  }
  return v11;
}
