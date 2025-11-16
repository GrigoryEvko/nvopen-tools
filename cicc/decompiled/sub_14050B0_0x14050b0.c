// Function: sub_14050B0
// Address: 0x14050b0
//
__int64 __fastcall sub_14050B0(__int64 a1, unsigned __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rcx
  __int64 v9; // rsi
  __int64 i; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 j; // r13
  __int64 v14; // rdi
  __int64 *v15; // r13
  __int64 v16; // r12
  unsigned int v17; // ebx
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  __int64 k; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  int v24; // ecx
  __int64 v25; // rcx
  const char *v26; // r9
  _QWORD *v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r8
  __int64 (__fastcall *v34)(__int64, __int64); // rax
  __int64 v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rcx
  __int64 v39; // r8
  __int64 v40; // rax
  __int64 v41; // rdx
  char v42; // al
  __int64 v43; // rdi
  const char *v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rdx
  unsigned int m; // r13d
  __int64 v54; // rdx
  __int64 v55; // rax
  int v56; // r13d
  unsigned int v57; // ebx
  unsigned int v58; // eax
  __int64 (*v59)(); // rax
  __int64 v60; // rax
  __int64 *v61; // r15
  __int64 v62; // r12
  _QWORD *v63; // rbx
  __int64 v64; // rcx
  unsigned __int64 v65; // rsi
  _QWORD *v66; // rax
  char v67; // r11
  _BOOL4 v68; // r11d
  __int64 v69; // rax
  __int64 *v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rax
  unsigned int v76; // [rsp+Ch] [rbp-B4h]
  _QWORD *v77; // [rsp+10h] [rbp-B0h]
  _QWORD *v78; // [rsp+18h] [rbp-A8h]
  __int64 v79; // [rsp+20h] [rbp-A0h]
  __int64 v80; // [rsp+28h] [rbp-98h]
  __int64 v81; // [rsp+28h] [rbp-98h]
  __int64 v82; // [rsp+30h] [rbp-90h]
  char v83; // [rsp+30h] [rbp-90h]
  __int64 v84; // [rsp+30h] [rbp-90h]
  __int64 *v85; // [rsp+38h] [rbp-88h]
  __int64 v86; // [rsp+38h] [rbp-88h]
  unsigned __int8 v87; // [rsp+40h] [rbp-80h]
  _BOOL4 v88; // [rsp+40h] [rbp-80h]
  __int64 v89; // [rsp+40h] [rbp-80h]
  __int64 *v90; // [rsp+48h] [rbp-78h]
  unsigned int v91; // [rsp+48h] [rbp-78h]
  __int64 *v92; // [rsp+48h] [rbp-78h]
  __int64 v93; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v94[12]; // [rsp+60h] [rbp-60h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_98:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9920C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_98;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9920C);
  v7 = *(_QWORD *)(a1 + 176);
  v8 = (_QWORD *)(a1 + 328);
  v78 = (_QWORD *)v6;
  *(_QWORD *)(a1 + 648) = v6 + 160;
  v9 = *(_QWORD *)(v7 + 8);
  v77 = *(_QWORD **)(a2 + 40);
  for ( i = *(_QWORD *)(v7 + 16); i != v9; *(v8 - 1) = v7 )
  {
    v11 = *(_QWORD *)(i - 8);
    i -= 8;
    ++v8;
    v7 = v11 + 224;
  }
  v12 = v78[24];
  for ( j = v78[25]; v12 != j; j -= 8 )
  {
    v14 = *(_QWORD *)(j - 8);
    v9 = a1 + 568;
    sub_1404F70(v14, (__int64 *)(a1 + 568));
  }
  v15 = *(__int64 **)(a1 + 584);
  v85 = *(__int64 **)(a1 + 616);
  if ( v85 != v15 )
  {
    v87 = 0;
    v90 = *(__int64 **)(a1 + 600);
    v82 = *(_QWORD *)(a1 + 608);
    do
    {
      v16 = *v15;
      if ( *(_DWORD *)(a1 + 192) )
      {
        v17 = 0;
        do
        {
          while ( 1 )
          {
            v7 = *(_QWORD *)(a1 + 184);
            v18 = *(_QWORD *)(v7 + 8LL * v17);
            v19 = *(__int64 (**)())(*(_QWORD *)v18 + 152LL);
            if ( v19 != sub_13C9000 )
              break;
            if ( ++v17 >= *(_DWORD *)(a1 + 192) )
              goto LABEL_16;
          }
          v9 = v16;
          ++v17;
          v87 |= ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD *))v19)(v18, v16, a1, v8);
        }
        while ( v17 < *(_DWORD *)(a1 + 192) );
      }
LABEL_16:
      if ( v90 == ++v15 )
      {
        v15 = *(__int64 **)(v82 + 8);
        v82 += 8;
        v8 = v15 + 64;
        v90 = v15 + 64;
      }
    }
    while ( v85 != v15 );
    v20 = sub_16033E0(*v77, v9, v7, v8);
    v76 = 0;
    v83 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v20 + 24LL))(v20, "size-info", 9);
    for ( k = *(_QWORD *)(a1 + 616); *(_QWORD *)(a1 + 584) != k; *(_QWORD *)(a1 + 616) = k )
    {
      v22 = *(_QWORD *)(a1 + 624);
      *(_BYTE *)(a1 + 664) = 0;
      if ( v22 == k )
        k = *(_QWORD *)(*(_QWORD *)(a1 + 640) - 8LL) + 512LL;
      v23 = *(_QWORD *)(k - 8);
      v24 = *(_DWORD *)(a1 + 192);
      *(_QWORD *)(a1 + 656) = v23;
      if ( v24 )
      {
        v91 = 0;
        while ( 1 )
        {
          v27 = *(_QWORD **)(*(_QWORD *)(a1 + 184) + 8LL * v91);
          v28 = sub_1649960(**(_QWORD **)(v23 + 32));
          sub_160F160(a1 + 160, v27, 0, 7, v28, v29);
          sub_1615D60(a1 + 160, v27);
          sub_1614C80(a1 + 160, v27);
          v30 = **(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL);
          sub_16C6860(v94);
          v94[2] = v27;
          v94[3] = v30;
          v94[0] = &unk_49ED7C0;
          v94[4] = 0;
          v31 = sub_1612E30(v27);
          v32 = v31;
          if ( v31 )
            sub_16D7910(v31);
          sub_1403F30(&v93, v27, *(_QWORD *)(a1 + 168));
          if ( v83 )
            v76 = sub_160E760(a1 + 160, v77);
          v33 = *(_QWORD *)(a1 + 656);
          v34 = *(__int64 (__fastcall **)(__int64, __int64))(*v27 + 144LL);
          if ( v34 == sub_1403A20 )
          {
            v80 = *(_QWORD *)(a1 + 656);
            v35 = *(_QWORD *)(v33 + 40);
            v36 = sub_1403960(*(_QWORD **)(v33 + 32), v35);
            if ( v38 != v36 )
            {
              v40 = sub_1649960(*(_QWORD *)(*v36 + 56LL));
              v35 = v41;
              v42 = sub_160E740(v40, v41);
              v39 = v80;
              if ( v42 )
              {
                v35 = v27[20];
                sub_13FC6E0(v80, v35, (__int64)(v27 + 21));
              }
            }
          }
          else
          {
            v35 = *(_QWORD *)(a1 + 656);
            v87 |= ((__int64 (__fastcall *)(_QWORD *, __int64, __int64))v34)(v27, v35, a1);
          }
          if ( v83 )
          {
            v35 = (__int64)v27;
            sub_160FF80(a1 + 160, v27, v77, v76, v39);
          }
          v43 = v93;
          if ( v93 )
          {
            if ( v87 )
            {
              v35 = 2;
              (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD *, __int64))(*(_QWORD *)v93 + 56LL))(
                v93,
                2,
                v37,
                v38,
                v39);
              v43 = v93;
            }
            (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD *, __int64))(*(_QWORD *)v43 + 48LL))(
              v43,
              v35,
              v37,
              v38,
              v39);
          }
          if ( v32 )
            sub_16D7950(v32, v35, v37);
          v94[0] = &unk_49ED7C0;
          nullsub_616(v94, v35, v37, v38, v39);
          if ( v87 )
          {
            v44 = "<deleted loop>";
            v45 = 14;
            if ( !*(_BYTE *)(a1 + 664) )
            {
              v45 = 14;
              v44 = "<unnamed loop>";
              v51 = **(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL);
              if ( v51 )
              {
                if ( (*(_BYTE *)(v51 + 23) & 0x20) != 0 )
                {
                  v44 = (const char *)sub_1649960(v51);
                  v45 = v52;
                }
              }
            }
            sub_160F160(a1 + 160, v27, 1, 7, v44, v45);
          }
          sub_1615E90(a1 + 160, v27);
          if ( *(_BYTE *)(a1 + 664) )
          {
            sub_1404680(a1, *(_QWORD *)(a1 + 656));
          }
          else
          {
            v46 = sub_1612E30(v78);
            v47 = v46;
            if ( v46 )
            {
              sub_16D7910(v46);
              sub_1403F30(v94, v78, *(_QWORD *)(a1 + 168));
              nullsub_529();
              if ( v94[0] )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v94[0] + 48LL))(v94[0]);
              sub_16D7950(v47, v78, v48);
            }
            else
            {
              sub_1403F30(v94, v78, *(_QWORD *)(a1 + 168));
              nullsub_529();
              if ( v94[0] )
                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v94[0] + 48LL))(v94[0]);
            }
            nullsub_568(a1 + 160, v27);
            v49 = sub_15E0530(a2);
            sub_16027A0(v49);
          }
          sub_16145F0(a1 + 160, v27);
          sub_16176C0(a1 + 160, v27);
          v25 = 9;
          v26 = "<deleted>";
          if ( !*(_BYTE *)(a1 + 664) )
          {
            v26 = (const char *)sub_1649960(**(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL));
            v25 = v50;
          }
          sub_1615450(a1 + 160, v27, v26, v25, 7);
          if ( *(_BYTE *)(a1 + 664) )
            break;
          if ( ++v91 >= *(_DWORD *)(a1 + 192) )
            goto LABEL_63;
          v23 = *(_QWORD *)(a1 + 656);
        }
        for ( m = 0; m < *(_DWORD *)(a1 + 192); ++m )
        {
          v54 = m;
          sub_16151B0(a1 + 160, *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v54), "<deleted>", 9, 7);
        }
LABEL_63:
        v22 = *(_QWORD *)(a1 + 624);
      }
      v55 = *(_QWORD *)(a1 + 616);
      if ( v22 == v55 )
      {
        j_j___libc_free_0(v22, 512);
        v71 = (__int64 *)(*(_QWORD *)(a1 + 640) - 8LL);
        *(_QWORD *)(a1 + 640) = v71;
        v72 = *v71;
        v73 = *v71 + 512;
        *(_QWORD *)(a1 + 624) = v72;
        k = v72 + 504;
        *(_QWORD *)(a1 + 632) = v73;
      }
      else
      {
        k = v55 - 8;
      }
    }
    if ( *(_DWORD *)(a1 + 192) )
    {
      v56 = v87;
      v57 = 0;
      do
      {
        while ( 1 )
        {
          v59 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 184) + 8LL * v57) + 160LL);
          if ( v59 != sub_13C9010 )
            break;
          v58 = *(_DWORD *)(a1 + 192);
          if ( ++v57 >= v58 )
            goto LABEL_72;
        }
        ++v57;
        v56 |= v59();
        v58 = *(_DWORD *)(a1 + 192);
      }
      while ( v57 < v58 );
LABEL_72:
      if ( (_BYTE)v56 )
      {
        v84 = 0;
        v81 = 8LL * v58;
        if ( v58 )
        {
          v79 = a1;
          while ( 1 )
          {
            v60 = *(_QWORD *)(*(_QWORD *)(v79 + 184) + v84);
            v61 = *(__int64 **)(v60 + 32);
            v92 = *(__int64 **)(v60 + 40);
            if ( v61 != v92 )
              break;
LABEL_87:
            v84 += 8;
            if ( v81 == v84 )
              return 1;
          }
          while ( 1 )
          {
            v62 = *v61;
            v63 = *(_QWORD **)(*v61 + 72);
            v64 = *v61 + 64;
            if ( !v63 )
              break;
            while ( 1 )
            {
              v65 = v63[4];
              v66 = (_QWORD *)v63[3];
              v67 = 0;
              if ( a2 < v65 )
              {
                v66 = (_QWORD *)v63[2];
                v67 = v56;
              }
              if ( !v66 )
                break;
              v63 = v66;
            }
            if ( v67 )
            {
              if ( *(_QWORD **)(v62 + 80) != v63 )
                goto LABEL_91;
LABEL_84:
              v68 = 1;
              if ( (_QWORD *)v64 == v63 )
              {
LABEL_85:
                v86 = v64;
                v88 = v68;
                v69 = sub_22077B0(40);
                *(_QWORD *)(v69 + 32) = a2;
                sub_220F040(v88, v69, v63, v86);
                ++*(_QWORD *)(v62 + 96);
                goto LABEL_86;
              }
LABEL_93:
              v68 = a2 < v63[4];
              goto LABEL_85;
            }
            if ( v65 < a2 )
              goto LABEL_84;
LABEL_86:
            if ( v92 == ++v61 )
              goto LABEL_87;
          }
          v63 = (_QWORD *)(*v61 + 64);
          if ( v64 == *(_QWORD *)(v62 + 80) )
          {
            v68 = 1;
            goto LABEL_85;
          }
LABEL_91:
          v89 = *v61 + 64;
          v74 = sub_220EF80(v63);
          v64 = v89;
          if ( a2 <= *(_QWORD *)(v74 + 32) )
            goto LABEL_86;
          v68 = 1;
          if ( (_QWORD *)v89 == v63 )
            goto LABEL_85;
          goto LABEL_93;
        }
        return 1;
      }
    }
    else if ( v87 )
    {
      return 1;
    }
  }
  return 0;
}
