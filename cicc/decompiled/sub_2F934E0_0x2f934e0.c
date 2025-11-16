// Function: sub_2F934E0
// Address: 0x2f934e0
//
_QWORD *__fastcall sub_2F934E0(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *result; // rax
  unsigned __int64 v8; // r9
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // r13
  unsigned int v12; // r12d
  unsigned int v13; // edi
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned int v19; // eax
  unsigned __int64 v20; // rcx
  int v21; // eax
  unsigned __int64 v22; // r8
  __int64 v23; // rdx
  __int64 *v24; // rdx
  void (*v25)(); // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 (*v28)(); // rax
  __int64 v29; // rsi
  unsigned int v30; // eax
  char v31; // al
  int v32; // edx
  __int64 v33; // rdx
  __int64 v34; // r12
  int v35; // ebx
  __int64 v36; // rdx
  __int64 *v37; // r14
  unsigned int v38; // r15d
  __int16 *v39; // r13
  __int64 (__fastcall *v40)(__int64, __int64, unsigned int); // rax
  __int64 (__fastcall *v41)(__int64, __int64, unsigned int); // rax
  int v42; // edx
  __int64 *v43; // r15
  unsigned __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rbx
  int v47; // r12d
  __int64 v48; // rdx
  unsigned int v49; // r11d
  __int16 *v50; // r14
  unsigned int v51; // ecx
  unsigned int v52; // edi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  unsigned int v56; // r9d
  __int64 v57; // rcx
  __int64 v58; // rcx
  _DWORD *v59; // rsi
  unsigned int v60; // r10d
  __int64 v61; // r8
  unsigned int v62; // r12d
  __int64 v63; // r9
  unsigned int v64; // ebx
  unsigned int v65; // edx
  __int64 v66; // rcx
  int v67; // edx
  unsigned int v68; // ebx
  unsigned int v69; // edx
  __int64 v70; // rcx
  __int64 v71; // r13
  __int64 v72; // rax
  __int64 v73; // [rsp+0h] [rbp-90h]
  unsigned int v74; // [rsp+Ch] [rbp-84h]
  __int64 v75; // [rsp+10h] [rbp-80h]
  __int64 v76; // [rsp+18h] [rbp-78h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  __int16 *v78; // [rsp+28h] [rbp-68h]
  unsigned __int64 v79; // [rsp+30h] [rbp-60h]
  int v81; // [rsp+3Ch] [rbp-54h]
  __int64 v82; // [rsp+48h] [rbp-48h]
  unsigned __int64 v83; // [rsp+48h] [rbp-48h]
  unsigned __int64 v84; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v85; // [rsp+58h] [rbp-38h]
  int v86; // [rsp+5Ch] [rbp-34h]

  v75 = *(_QWORD *)a2;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 32LL) + 40LL * a3;
  v6 = *(unsigned int *)(v5 + 8);
  v77 = v5;
  result = (_QWORD *)sub_2EBF3A0(*(_QWORD **)(a1 + 40), v6);
  if ( (_BYTE)result )
    return result;
  v9 = *(_QWORD *)(a1 + 24);
  v82 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v10 = *(_BYTE *)(v5 + 3);
  v11 = a1 + 328;
  v81 = 2 - ((v10 & 0x10) == 0);
  v79 = ((-(__int64)((v10 & 0x10) == 0) & 0xFFFFFFFFFFFFFFFELL) + 4) | a2 & 0xFFFFFFFFFFFFFFF9LL;
  v73 = 24 * v6;
  v12 = *(_DWORD *)(*(_QWORD *)(v9 + 8) + 24 * v6 + 16) & 0xFFF;
  v78 = (__int16 *)(*(_QWORD *)(v9 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v9 + 8) + 24 * v6 + 16) >> 12));
  v76 = a1 + 600;
  while ( v78 )
  {
    v13 = *(_DWORD *)(a1 + 976);
    v14 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1176) + 2LL * v12);
    if ( v14 >= v13 )
      goto LABEL_27;
    v15 = *(_QWORD *)(a1 + 968);
    while ( 1 )
    {
      v16 = v14;
      v17 = v15 + 24LL * v14;
      if ( v12 == *(_DWORD *)(v17 + 12) )
      {
        v18 = *(unsigned int *)(v17 + 16);
        if ( (_DWORD)v18 != -1 && *(_DWORD *)(v15 + 24 * v18 + 20) == -1 )
          break;
      }
      v14 += 0x10000;
      if ( v13 <= v14 )
        goto LABEL_27;
    }
    if ( v14 != -1 )
    {
      v74 = v12;
      while ( 1 )
      {
        v26 = 24 * v16;
        v24 = (__int64 *)(v15 + 24 * v16);
        v27 = *v24;
        if ( *v24 == v11 )
          goto LABEL_18;
        v28 = *(__int64 (**)())(*(_QWORD *)a1 + 128LL);
        if ( v28 == sub_2EC0A10
          || (v31 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v28)(a1, a2, *v24),
              v15 = *(_QWORD *)(a1 + 968),
              v24 = (__int64 *)(v15 + v26),
              v31) )
        {
          if ( a2 != v27 )
          {
            v22 = *(_QWORD *)v27;
            v29 = *(_QWORD *)(*(_QWORD *)v27 + 32LL) + 40LL * *((unsigned int *)v24 + 2);
            if ( v81 != 2 )
            {
              v30 = *(_DWORD *)(v29 + 8);
              v86 = 0;
              v84 = v79;
              v85 = v30;
              goto LABEL_15;
            }
            if ( (((*(_BYTE *)(v77 + 3) & 0x10) != 0) & (*(_BYTE *)(v77 + 3) >> 6)) == 0
              || (((*(_BYTE *)(v29 + 3) & 0x10) != 0) & (*(_BYTE *)(v29 + 3) >> 6)) == 0 )
            {
              v19 = *(_DWORD *)(v29 + 8);
              v20 = *(_QWORD *)v27;
              v86 = 0;
              v84 = v79;
              v85 = v19;
              v21 = sub_2FF8480(v76, v75, a3, v20);
              v23 = *(_QWORD *)(a1 + 968);
              v86 = v21;
              v24 = (__int64 *)(v26 + v23);
LABEL_15:
              v25 = *(void (**)())(*(_QWORD *)v82 + 344LL);
              if ( v25 != nullsub_1667 )
                ((void (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, _QWORD, unsigned __int64 *, __int64))v25)(
                  v82,
                  a2,
                  a3,
                  v27,
                  *((unsigned int *)v24 + 2),
                  &v84,
                  v76);
              sub_2F8F1B0(v27, (__int64)&v84, 1u, v15, v22, v8);
              v15 = *(_QWORD *)(a1 + 968);
              v24 = (__int64 *)(v15 + v26);
            }
          }
LABEL_18:
          v16 = *((unsigned int *)v24 + 5);
          if ( (_DWORD)v16 == -1 )
            goto LABEL_26;
        }
        else
        {
          v16 = *((unsigned int *)v24 + 5);
          if ( (_DWORD)v16 == -1 )
          {
LABEL_26:
            v12 = v74;
            break;
          }
        }
      }
    }
LABEL_27:
    v32 = *v78++;
    v12 += v32;
    if ( !(_WORD)v32 )
      break;
  }
  if ( (*(_BYTE *)(v77 + 3) & 0x10) == 0 )
  {
    *(_BYTE *)(a2 + 248) |= 0x20u;
    v33 = *(_QWORD *)(a1 + 24);
    v34 = *(_QWORD *)(v33 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v33 + 8) + v73 + 16) >> 12);
    result = &v84;
    v35 = *(_DWORD *)(*(_QWORD *)(v33 + 8) + v73 + 16) & 0xFFF;
    do
    {
      if ( !v34 )
        break;
      v86 = v35;
      v84 = a2;
      v34 += 2;
      v85 = a3;
      result = (_QWORD *)sub_2F932E0(a1 + 1200, (__int64)&v84);
      v35 += *(__int16 *)(v34 - 2);
    }
    while ( *(_WORD *)(v34 - 2) );
    if ( *(_BYTE *)(a1 + 896) )
    {
      result = (_QWORD *)v77;
      *(_BYTE *)(v77 + 3) &= ~0x40u;
    }
    return result;
  }
  sub_2F91990(a1, a2, a3);
  v36 = *(_QWORD *)(a1 + 24);
  v83 = a2;
  v37 = (__int64 *)a1;
  v38 = *(_DWORD *)(*(_QWORD *)(v36 + 8) + v73 + 16) & 0xFFF;
  v39 = (__int16 *)(*(_QWORD *)(v36 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v36 + 8) + v73 + 16) >> 12));
  do
  {
    if ( !v39 )
      break;
    v40 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*v37 + 136);
    if ( v40 == sub_2ECA670 )
    {
      LODWORD(v84) = v38;
      sub_2ECA430(v37 + 150, (unsigned int *)&v84);
    }
    else
    {
      v40((__int64)v37, v83, v38);
    }
    if ( (((*(_BYTE *)(v77 + 3) & 0x10) != 0) & (*(_BYTE *)(v77 + 3) >> 6)) == 0 )
    {
      v41 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*v37 + 144);
      if ( v41 == sub_2ECA650 )
      {
        LODWORD(v84) = v38;
        sub_2ECA430(v37 + 121, (unsigned int *)&v84);
      }
      else
      {
        v41((__int64)v37, v83, v38);
      }
    }
    v42 = *v39++;
    v38 += v42;
  }
  while ( (_WORD)v42 );
  v43 = v37;
  v44 = v83;
  if ( (((*(_BYTE *)(v77 + 3) & 0x10) != 0) & (*(_BYTE *)(v77 + 3) >> 6)) != 0 && (*(_BYTE *)(v83 + 248) & 2) != 0 )
  {
    v48 = v43[3];
    v49 = *(_DWORD *)(*(_QWORD *)(v48 + 8) + v73 + 16) & 0xFFF;
    v50 = (__int16 *)(*(_QWORD *)(v48 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v48 + 8) + v73 + 16) >> 12));
    while ( 1 )
    {
      if ( !v50 )
      {
LABEL_81:
        v44 = v83;
        goto LABEL_45;
      }
      v51 = *((_DWORD *)v43 + 244);
      v52 = *(unsigned __int16 *)(v43[147] + 2LL * v49);
      if ( v52 < v51 )
      {
        v53 = v43[121];
        while ( 1 )
        {
          v54 = v53 + 24LL * v52;
          if ( v49 == *(_DWORD *)(v54 + 12) )
          {
            v55 = *(unsigned int *)(v54 + 16);
            if ( (_DWORD)v55 != -1 && *(_DWORD *)(v53 + 24 * v55 + 20) == -1 )
              break;
          }
          v52 += 0x10000;
          if ( v51 <= v52 )
            goto LABEL_80;
        }
        if ( v52 != -1 )
          break;
      }
LABEL_80:
      v67 = *v50++;
      v49 += v67;
      if ( !(_WORD)v67 )
        goto LABEL_81;
    }
    v56 = v49;
    v57 = 0xFFFFFFFFLL;
    while ( (_DWORD)v57 == -1 )
    {
      v64 = *((_DWORD *)v43 + 244);
      v65 = *(unsigned __int16 *)(v43[147] + 2LL * v56);
      if ( v65 < v64 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v66 = v53 + 24LL * v65;
            if ( v56 == *(_DWORD *)(v66 + 12) )
            {
              v60 = *(_DWORD *)(v66 + 16);
              if ( v60 != -1 )
                break;
            }
            v65 += 0x10000;
            if ( v64 <= v65 )
              goto LABEL_79;
          }
          v61 = 24LL * v60;
          v59 = (_DWORD *)(v53 + v61);
          if ( *(_DWORD *)(v53 + v61 + 20) == -1 )
            break;
          v65 += 0x10000;
          if ( v64 <= v65 )
            goto LABEL_79;
        }
LABEL_68:
        if ( (*(_BYTE *)(*(_QWORD *)v59 + 248LL) & 2) == 0 )
          goto LABEL_80;
        goto LABEL_69;
      }
LABEL_79:
      v60 = *(_DWORD *)(v53 + 0x17FFFFFFF8LL);
      v61 = 24LL * v60;
      v59 = (_DWORD *)(v53 + v61);
      if ( (*(_BYTE *)(*(_QWORD *)(v53 + v61) + 248LL) & 2) == 0 )
        goto LABEL_80;
LABEL_69:
      v62 = v59[4];
      v63 = 24LL * v62;
      if ( (_DWORD *)(v53 + v63) == v59 )
      {
        v56 = v59[3];
        v57 = 0xFFFFFFFFLL;
      }
      else
      {
        v58 = (unsigned int)v59[5];
        if ( *(_DWORD *)(v53 + v63 + 20) == -1 )
        {
          *(_WORD *)(v43[147] + 2LL * (unsigned int)v59[3]) = v58;
          *(_DWORD *)(v43[121] + 24LL * (unsigned int)v59[5] + 16) = v59[4];
          v56 = v59[3];
          v57 = (unsigned int)v59[5];
          v59 = (_DWORD *)(v61 + v43[121]);
        }
        else if ( (_DWORD)v58 == -1 )
        {
          v68 = *((_DWORD *)v43 + 244);
          v69 = *(unsigned __int16 *)(v43[147] + 2LL * (unsigned int)v59[3]);
          if ( v69 < v68 )
          {
            while ( 1 )
            {
              v70 = v53 + 24LL * v69;
              if ( v59[3] == *(_DWORD *)(v70 + 12) )
              {
                v71 = *(unsigned int *)(v70 + 16);
                if ( (_DWORD)v71 != -1 && *(_DWORD *)(v53 + 24 * v71 + 20) == -1 )
                  break;
              }
              v69 += 0x10000;
              if ( v68 <= v69 )
                goto LABEL_89;
            }
          }
          else
          {
LABEL_89:
            v70 = v53 + 0x17FFFFFFE8LL;
          }
          *(_DWORD *)(v70 + 16) = v62;
          *(_DWORD *)(v43[121] + 24LL * (unsigned int)v59[4] + 20) = v59[5];
          v72 = v43[121];
          v56 = v59[3];
          v57 = *(unsigned int *)(v72 + 24LL * (unsigned int)v59[4] + 20);
          v59 = (_DWORD *)(v72 + v61);
        }
        else
        {
          *(_DWORD *)(v53 + 24 * v58 + 16) = v62;
          v57 = (unsigned int)v59[5];
          *(_DWORD *)(v43[121] + v63 + 20) = v57;
          v56 = v59[3];
          v59 = (_DWORD *)(v61 + v43[121]);
        }
      }
      v59[4] = -1;
      *(_DWORD *)(v43[121] + v61 + 20) = *((_DWORD *)v43 + 298);
      *((_DWORD *)v43 + 298) = v60;
      ++*((_DWORD *)v43 + 299);
      if ( v60 == v52 )
        goto LABEL_80;
      v53 = v43[121];
    }
    v60 = *(_DWORD *)(v53 + 24 * v57 + 16);
    v61 = 24LL * v60;
    v59 = (_DWORD *)(v53 + v61);
    goto LABEL_68;
  }
LABEL_45:
  v45 = v43[3];
  v46 = *(_QWORD *)(v45 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v45 + 8) + v73 + 16) >> 12);
  result = &v84;
  v47 = *(_DWORD *)(*(_QWORD *)(v45 + 8) + v73 + 16) & 0xFFF;
  do
  {
    if ( !v46 )
      break;
    v86 = v47;
    v46 += 2;
    v84 = v44;
    v85 = a3;
    result = (_QWORD *)sub_2F932E0((__int64)(v43 + 121), (__int64)&v84);
    v47 += *(__int16 *)(v46 - 2);
  }
  while ( *(_WORD *)(v46 - 2) );
  return result;
}
