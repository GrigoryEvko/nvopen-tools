// Function: sub_3387170
// Address: 0x3387170
//
void __fastcall sub_3387170(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // ecx
  __int64 v6; // rsi
  int v7; // ecx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // r8
  unsigned int *v12; // r15
  __int64 *v13; // rsi
  int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _DWORD *v20; // rsi
  int v21; // ecx
  bool v22; // zf
  unsigned __int8 v23; // r9
  char v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 *v29; // rsi
  __int64 v30; // r14
  unsigned int v31; // r15d
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // r14
  _QWORD *v35; // r15
  __int64 v36; // rdi
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 *v39; // rsi
  unsigned __int8 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 *v45; // rsi
  char *v46; // rcx
  __int64 v47; // r8
  __int64 *v48; // rdi
  _DWORD *v49; // r8
  int v50; // ecx
  char v51; // al
  _DWORD *v52; // rsi
  int v53; // ecx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rsi
  int v58; // r13d
  _DWORD *v59; // rsi
  int v60; // r12d
  unsigned __int8 *v61; // r8
  unsigned __int8 **v62; // r11
  unsigned __int8 *v63; // r8
  unsigned __int8 *v64; // r8
  int v65; // eax
  int v66; // r8d
  __int64 v67; // r9
  unsigned int v68; // r10d
  int v69; // r8d
  int v70; // r11d
  int v71; // [rsp+0h] [rbp-100h]
  unsigned __int8 v72; // [rsp+4h] [rbp-FCh]
  int v74; // [rsp+10h] [rbp-F0h]
  char v75; // [rsp+10h] [rbp-F0h]
  int v76; // [rsp+10h] [rbp-F0h]
  unsigned int *v77; // [rsp+28h] [rbp-D8h]
  bool v78; // [rsp+28h] [rbp-D8h]
  __int64 v79; // [rsp+30h] [rbp-D0h]
  _QWORD *v80; // [rsp+38h] [rbp-C8h]
  __int64 v81; // [rsp+40h] [rbp-C0h] BYREF
  _DWORD *v82; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v83[2]; // [rsp+50h] [rbp-B0h] BYREF
  _DWORD *v84; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v85; // [rsp+68h] [rbp-98h]
  _BYTE v86[32]; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v87; // [rsp+90h] [rbp-70h] BYREF
  __int64 v88; // [rsp+98h] [rbp-68h]
  _BYTE v89[96]; // [rsp+A0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 32LL);
  v80 = (_QWORD *)v3;
  if ( v3 )
  {
    v4 = *(_DWORD *)(v3 + 136);
    v6 = *(_QWORD *)(v3 + 120);
    if ( v4 )
    {
      v7 = v4 - 1;
      v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      v11 = v9;
      if ( a2 == *v9 )
      {
LABEL_4:
        v77 = (unsigned int *)v80[7];
        v12 = &v77[8 * *((unsigned int *)v11 + 2)];
      }
      else
      {
        v67 = *v9;
        v68 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v69 = 1;
        while ( v67 != -4096 )
        {
          v70 = v69 + 1;
          v68 = v7 & (v69 + v68);
          v11 = (__int64 *)(v6 + 16LL * v68);
          v67 = *v11;
          if ( a2 == *v11 )
            goto LABEL_4;
          v69 = v70;
        }
        v12 = (unsigned int *)v80[7];
        v77 = v12;
        v8 = v7 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
      }
      if ( a2 == v10 )
      {
LABEL_6:
        v77 += 8 * *((unsigned int *)v9 + 3);
      }
      else
      {
        v65 = 1;
        while ( v10 != -4096 )
        {
          v66 = v65 + 1;
          v8 = v7 & (v65 + v8);
          v9 = (__int64 *)(v6 + 16LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_6;
          v65 = v66;
        }
      }
      if ( v77 != v12 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(*v80 + 40LL * *v12);
          sub_3382030(a1, v15, *((_QWORD *)v12 + 1));
          v16 = *((_QWORD *)v12 + 3);
          if ( *(_BYTE *)v16 == 4 )
          {
            if ( !*(_DWORD *)(v16 + 144) && !(unsigned __int8)sub_AF4500(*((_QWORD *)v12 + 1)) )
              goto LABEL_10;
LABEL_17:
            sub_B58DC0(&v87, (unsigned __int8 **)v12 + 3);
            if ( sub_3366350((__int64 *)&v87) )
              goto LABEL_10;
            sub_B58DC0(&v84, (unsigned __int8 **)v12 + 3);
            v87 = (__int64 *)v89;
            v88 = 0x600000000LL;
            v83[0] = v85;
            v82 = v84;
            sub_F388A0((__int64)&v87, (__int64 *)&v82, v83, v17, v18, v19);
            v20 = (_DWORD *)*((_QWORD *)v12 + 2);
            v21 = *(_DWORD *)(a1 + 848);
            v22 = **((_BYTE **)v12 + 3) == 4;
            v84 = v20;
            v23 = v22;
            if ( v20 )
            {
              v71 = v21;
              v72 = v22;
              sub_B96E90((__int64)&v84, (__int64)v20, 1);
              v21 = v71;
              v23 = v72;
            }
            v24 = sub_3380DB0(a1, v87, (unsigned int)v88, v15, *((_QWORD *)v12 + 1), &v84, v21, v23);
            if ( v84 )
              sub_B91220((__int64)&v84, (__int64)v84);
            if ( !v24 )
            {
              sub_B58DC0(v83, (unsigned __int8 **)v12 + 3);
              v85 = 0x400000000LL;
              v84 = v86;
              v82 = (_DWORD *)v83[1];
              v81 = v83[0];
              sub_F388A0((__int64)&v84, &v81, &v82, v54, v55, v56);
              v57 = *((_QWORD *)v12 + 2);
              v58 = *(_DWORD *)(a1 + 848);
              v83[0] = v57;
              if ( v57 )
                sub_B96E90((__int64)v83, v57, 1);
              sub_3386E40(
                a1,
                (__int64)&v84,
                *(_QWORD *)(*v80 + 40LL * *v12),
                *((_QWORD *)v12 + 1),
                (unsigned int)v85 > 1,
                v83,
                v58);
              if ( v83[0] )
                sub_B91220((__int64)v83, v83[0]);
              if ( v84 != (_DWORD *)v86 )
                _libc_free((unsigned __int64)v84);
            }
            if ( v87 == (__int64 *)v89 )
              goto LABEL_14;
            _libc_free((unsigned __int64)v87);
            v12 += 8;
            if ( v12 == v77 )
            {
LABEL_25:
              v80 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 32LL);
              break;
            }
          }
          else
          {
            if ( (unsigned __int8)(*(_BYTE *)v16 - 5) > 0x1Fu )
              goto LABEL_17;
LABEL_10:
            v13 = (__int64 *)*((_QWORD *)v12 + 2);
            v14 = *(_DWORD *)(a1 + 848);
            v87 = v13;
            if ( v13 )
              sub_B96E90((__int64)&v87, (__int64)v13, 1);
            sub_3382930(a1, v15, *((_QWORD **)v12 + 1), (_DWORD **)&v87, v14);
            if ( v87 )
              sub_B91220((__int64)&v87, (__int64)v87);
LABEL_14:
            v12 += 8;
            if ( v12 == v77 )
              goto LABEL_25;
          }
        }
      }
    }
  }
  v25 = *(_QWORD *)(a2 + 64);
  if ( v25 )
  {
    v26 = sub_B14240(v25);
    v79 = v27;
    if ( v27 != v26 )
    {
      v28 = v26;
      while ( 1 )
      {
        if ( *(_BYTE *)(v28 + 32) == 1 )
        {
          v29 = *(__int64 **)(v28 + 24);
          v30 = *(_QWORD *)(a1 + 864);
          v31 = *(_DWORD *)(a1 + 848);
          v87 = v29;
          if ( v29 )
            sub_B96E90((__int64)&v87, (__int64)v29, 1);
          v32 = sub_B11FB0(v28 + 40);
          v33 = sub_33E6440(v30, v32, &v87, v31);
          if ( v87 )
            sub_B91220((__int64)&v87, (__int64)v87);
          sub_33CF110(*(_QWORD *)(a1 + 864), v33);
          goto LABEL_34;
        }
        if ( v80 )
          goto LABEL_34;
        v34 = sub_B12000(v28 + 72);
        v35 = (_QWORD *)sub_B11F60(v28 + 80);
        sub_3382030(a1, v34, (__int64)v35);
        if ( *(_BYTE *)(v28 + 64) )
        {
          sub_B129C0(&v84, v28);
          v87 = (__int64 *)v89;
          v88 = 0x400000000LL;
          v83[0] = v85;
          v82 = v84;
          sub_FC7C70((__int64)&v87, (__int64 *)&v82, v83, v41, v42, v43);
          v44 = (unsigned int)v88;
          if ( !(_DWORD)v88 )
          {
            v59 = *(_DWORD **)(v28 + 24);
            v60 = *(_DWORD *)(a1 + 848);
            v84 = v59;
            if ( v59 )
              goto LABEL_73;
            goto LABEL_74;
          }
          v45 = v87;
          v46 = (char *)&v87[(unsigned int)v88];
          v47 = (8LL * (unsigned int)v88) >> 3;
          if ( (8LL * (unsigned int)v88) >> 5 )
          {
            v48 = v87;
            while ( *v48 && (unsigned int)*(unsigned __int8 *)*v48 - 12 > 1 )
            {
              v61 = (unsigned __int8 *)v48[1];
              v62 = (unsigned __int8 **)(v48 + 1);
              if ( !v61
                || (unsigned int)*v61 - 12 <= 1
                || (v63 = (unsigned __int8 *)v48[2], v62 = (unsigned __int8 **)(v48 + 2), !v63)
                || (unsigned int)*v63 - 12 <= 1
                || (v64 = (unsigned __int8 *)v48[3], v62 = (unsigned __int8 **)(v48 + 3), !v64)
                || (unsigned int)*v64 - 12 <= 1 )
              {
                if ( v46 == (char *)v62 )
                  goto LABEL_53;
                goto LABEL_81;
              }
              v48 += 4;
              if ( v48 == &v87[4 * ((8LL * (unsigned int)v88) >> 5)] )
              {
                v47 = (v46 - (char *)v48) >> 3;
                goto LABEL_89;
              }
            }
LABEL_52:
            if ( v46 == (char *)v48 )
            {
LABEL_53:
              v49 = *(_DWORD **)(v28 + 24);
              v50 = *(_DWORD *)(a1 + 848);
              v22 = **(_BYTE **)(v28 + 40) == 4;
              v84 = v49;
              v78 = v22;
              if ( v49 )
              {
                v74 = v50;
                sub_B96E90((__int64)&v84, (__int64)v49, 1);
                v45 = v87;
                v44 = (unsigned int)v88;
                v50 = v74;
              }
              v51 = sub_3380DB0(a1, v45, v44, v34, (__int64)v35, &v84, v50, v78);
              if ( v84 )
              {
                v75 = v51;
                sub_B91220((__int64)&v84, (__int64)v84);
                v51 = v75;
              }
              if ( !v51 )
              {
                v52 = *(_DWORD **)(v28 + 24);
                v53 = *(_DWORD *)(a1 + 848);
                v84 = v52;
                if ( v52 )
                {
                  v76 = v53;
                  sub_B96E90((__int64)&v84, (__int64)v52, 1);
                  v53 = v76;
                }
                sub_3386E40(a1, (__int64)&v87, v34, (__int64)v35, v78, (__int64 *)&v84, v53);
                if ( v84 )
LABEL_61:
                  sub_B91220((__int64)&v84, (__int64)v84);
              }
LABEL_62:
              if ( v87 != (__int64 *)v89 )
                _libc_free((unsigned __int64)v87);
              goto LABEL_34;
            }
LABEL_81:
            v59 = *(_DWORD **)(v28 + 24);
            v60 = *(_DWORD *)(a1 + 848);
            v84 = v59;
            if ( v59 )
LABEL_73:
              sub_B96E90((__int64)&v84, (__int64)v59, 1);
LABEL_74:
            sub_3382930(a1, v34, v35, &v84, v60);
            if ( v84 )
              goto LABEL_61;
            goto LABEL_62;
          }
          v48 = v87;
LABEL_89:
          if ( v47 != 2 )
          {
            if ( v47 != 3 )
            {
              if ( v47 != 1 )
                goto LABEL_53;
              goto LABEL_92;
            }
            if ( !*v48 || (unsigned int)*(unsigned __int8 *)*v48 - 12 <= 1 )
              goto LABEL_52;
            ++v48;
          }
          if ( !*v48 || (unsigned int)*(unsigned __int8 *)*v48 - 12 <= 1 )
            goto LABEL_52;
          ++v48;
LABEL_92:
          if ( !*v48 || (unsigned int)*(unsigned __int8 *)*v48 - 12 <= 1 )
            goto LABEL_52;
          goto LABEL_53;
        }
        v36 = *(_QWORD *)(a1 + 960);
        if ( *(_BYTE *)(v36 + 1020) )
        {
          v37 = *(_QWORD **)(v36 + 1000);
          v38 = &v37[*(unsigned int *)(v36 + 1012)];
          if ( v37 != v38 )
          {
            while ( *v37 != v28 )
            {
              if ( v38 == ++v37 )
                goto LABEL_42;
            }
            goto LABEL_34;
          }
LABEL_42:
          v39 = *(__int64 **)(v28 + 24);
          v87 = v39;
          if ( v39 )
            sub_B96E90((__int64)&v87, (__int64)v39, 1);
          v40 = (unsigned __int8 *)sub_B12A50(v28, 0);
          sub_3380BE0(a1, v40, v34, v35, (__int64)&v87);
          if ( !v87 )
            goto LABEL_34;
          sub_B91220((__int64)&v87, (__int64)v87);
          v28 = *(_QWORD *)(v28 + 8);
          if ( v28 == v79 )
            return;
        }
        else
        {
          if ( !sub_C8CA60(v36 + 992, v28) )
            goto LABEL_42;
LABEL_34:
          v28 = *(_QWORD *)(v28 + 8);
          if ( v28 == v79 )
            return;
        }
      }
    }
  }
}
