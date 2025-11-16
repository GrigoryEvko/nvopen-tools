// Function: sub_34EA620
// Address: 0x34ea620
//
char __fastcall sub_34EA620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6, char a7)
{
  __int64 v7; // r15
  char result; // al
  __int64 v12; // rdi
  __int64 (*v13)(); // r10
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned int v18; // r15d
  __int64 v19; // r13
  __int64 (*v20)(); // rdx
  __int64 v21; // rdi
  __int64 (*v22)(); // r9
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r13
  __int64 (*v26)(); // rdx
  __int64 v27; // rdi
  __int64 (*v28)(); // r9
  int v29; // eax
  __int64 v30; // rcx
  int v31; // r13d
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 *v34; // rdi
  __int64 v35; // rdx
  __int64 (*v36)(); // rdx
  int v37; // eax
  int v38; // eax
  __int64 v39; // rbx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 *v42; // rdi
  __int64 v43; // rdx
  __int64 (*v44)(); // rdx
  int v45; // eax
  int v46; // eax
  __int64 (__fastcall *v47)(); // rax
  __int64 (*v48)(); // rdx
  int v49; // eax
  __int64 v50; // r12
  unsigned __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 (*v54)(); // rdx
  int v55; // eax
  int v56; // eax
  __int64 (__fastcall *v57)(); // rax
  __int64 (*v58)(); // rdx
  int v59; // eax
  __int64 v60; // rdx
  unsigned int v61; // r9d
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 (*v64)(); // rcx
  unsigned int v65; // eax
  __int64 *v66; // [rsp+8h] [rbp-88h]
  __int64 v68; // [rsp+18h] [rbp-78h]
  __int64 (*v69)(); // [rsp+18h] [rbp-78h]
  __int64 (*v70)(); // [rsp+18h] [rbp-78h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+18h] [rbp-78h]
  __int64 v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+20h] [rbp-70h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  __int64 v78; // [rsp+20h] [rbp-70h]
  __int64 v79; // [rsp+20h] [rbp-70h]
  __int64 v80; // [rsp+20h] [rbp-70h]
  __int64 v81; // [rsp+20h] [rbp-70h]
  __int64 v82; // [rsp+20h] [rbp-70h]
  __int64 v83; // [rsp+20h] [rbp-70h]
  int v84; // [rsp+38h] [rbp-58h] BYREF
  int v85; // [rsp+3Ch] [rbp-54h] BYREF
  __int64 v86; // [rsp+40h] [rbp-50h] BYREF
  __int64 v87; // [rsp+48h] [rbp-48h] BYREF
  __int64 v88; // [rsp+50h] [rbp-40h] BYREF
  __int64 v89[7]; // [rsp+58h] [rbp-38h] BYREF

  v7 = a6;
  v66 = *(__int64 **)(*(_QWORD *)(a2 + 16) + 32LL);
  if ( (unsigned __int8)sub_B2D610(*v66, 18) )
  {
    v14 = *(_QWORD *)(a2 + 16);
    v84 = 0;
    v15 = *(_QWORD *)(v14 + 56);
    v85 = 0;
    v86 = v15;
    v16 = *(_QWORD *)(a3 + 16);
    v87 = *(_QWORD *)(v16 + 56);
    v88 = v14 + 48;
    v89[0] = v16 + 48;
    if ( !(unsigned __int8)sub_34E9A00(a1, &v86, &v87, &v88, v89, &v84, &v85, v14, v16, 1) )
      BUG();
    v17 = v86;
    v18 = 0;
    v19 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 56LL);
    if ( v86 != v19 )
    {
      v20 = sub_2E77FD0;
      do
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(a1 + 528);
          v22 = *(__int64 (**)())(*(_QWORD *)v21 + 168LL);
          v23 = -1;
          if ( v22 != v20 )
          {
            v69 = v20;
            v75 = v17;
            v23 = ((__int64 (__fastcall *)(__int64, __int64))v22)(v21, v19);
            v20 = v69;
            v17 = v75;
          }
          v18 += v23;
          if ( !v19 )
            BUG();
          if ( (*(_BYTE *)v19 & 4) == 0 )
            break;
          v19 = *(_QWORD *)(v19 + 8);
          if ( v17 == v19 )
            goto LABEL_14;
        }
        while ( (*(_BYTE *)(v19 + 44) & 8) != 0 )
          v19 = *(_QWORD *)(v19 + 8);
        v19 = *(_QWORD *)(v19 + 8);
      }
      while ( v17 != v19 );
    }
LABEL_14:
    v24 = v87;
    v25 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 56LL);
    if ( v25 != v87 )
    {
      v26 = sub_2E77FD0;
      do
      {
        while ( 1 )
        {
          v27 = *(_QWORD *)(a1 + 528);
          v28 = *(__int64 (**)())(*(_QWORD *)v27 + 168LL);
          v29 = -1;
          if ( v28 != v26 )
          {
            v70 = v26;
            v76 = v24;
            v29 = ((__int64 (__fastcall *)(__int64, __int64))v28)(v27, v25);
            v26 = v70;
            v24 = v76;
          }
          v18 += v29;
          if ( !v25 )
            BUG();
          if ( (*(_BYTE *)v25 & 4) == 0 )
            break;
          v25 = *(_QWORD *)(v25 + 8);
          if ( v24 == v25 )
            goto LABEL_22;
        }
        while ( (*(_BYTE *)(v25 + 44) & 8) != 0 )
          v25 = *(_QWORD *)(v25 + 8);
        v25 = *(_QWORD *)(v25 + 8);
      }
      while ( v24 != v25 );
    }
LABEL_22:
    v30 = v88;
    v31 = 0;
    v32 = *(_QWORD *)(a2 + 16) + 48LL;
    if ( v32 != v88 )
    {
      do
      {
        while ( 1 )
        {
          v38 = *(_DWORD *)(v30 + 44);
          if ( (v38 & 4) != 0 || (v38 & 8) == 0 )
          {
            v33 = (*(_QWORD *)(*(_QWORD *)(v30 + 16) + 24LL) >> 10) & 1LL;
          }
          else
          {
            v68 = v32;
            v74 = v30;
            LOBYTE(v33) = sub_2E88A90(v30, 1024, 1);
            v30 = v74;
            v32 = v68;
          }
          v34 = *(__int64 **)(a1 + 528);
          v35 = *v34;
          if ( (_BYTE)v33 && (*(_BYTE *)a2 & 0x10) != 0 && !a7 )
          {
            v47 = *(__int64 (__fastcall **)())(v35 + 448);
            if ( v47 == sub_2FDCF50 )
            {
              v48 = *(__int64 (**)())(v35 + 168);
              v49 = -1;
              if ( v48 != sub_2E77FD0 )
              {
                v73 = v32;
                v83 = v30;
                v49 = ((__int64 (__fastcall *)(__int64 *, __int64))v48)(v34, v30);
                v30 = v83;
                v32 = v73;
              }
            }
            else
            {
              v72 = v32;
              v81 = v30;
              v49 = ((__int64 (__fastcall *)(__int64 *, __int64))v47)(v34, v30);
              v32 = v72;
              v30 = v81;
            }
            v31 += v49;
          }
          else
          {
            v36 = *(__int64 (**)())(v35 + 168);
            v37 = -1;
            if ( v36 != sub_2E77FD0 )
            {
              v71 = v32;
              v78 = v30;
              v37 = ((__int64 (__fastcall *)(__int64 *, __int64))v36)(v34, v30);
              v32 = v71;
              v30 = v78;
            }
            v18 += v37;
          }
          if ( (*(_BYTE *)v30 & 4) == 0 )
            break;
          v30 = *(_QWORD *)(v30 + 8);
          if ( v32 == v30 )
            goto LABEL_45;
        }
        while ( (*(_BYTE *)(v30 + 44) & 8) != 0 )
          v30 = *(_QWORD *)(v30 + 8);
        v30 = *(_QWORD *)(v30 + 8);
      }
      while ( v32 != v30 );
    }
LABEL_45:
    v39 = v89[0];
    v40 = *(_QWORD *)(a3 + 16) + 48LL;
    if ( v89[0] != v40 )
    {
      do
      {
        while ( 1 )
        {
          v46 = *(_DWORD *)(v39 + 44);
          if ( (v46 & 4) != 0 || (v46 & 8) == 0 )
          {
            v41 = (*(_QWORD *)(*(_QWORD *)(v39 + 16) + 24LL) >> 10) & 1LL;
          }
          else
          {
            v77 = v40;
            LOBYTE(v41) = sub_2E88A90(v39, 1024, 1);
            v40 = v77;
          }
          v42 = *(__int64 **)(a1 + 528);
          v43 = *v42;
          if ( (_BYTE)v41 && (*(_BYTE *)a3 & 0x10) != 0 && !a7 )
          {
            v57 = *(__int64 (__fastcall **)())(v43 + 448);
            if ( v57 == sub_2FDCF50 )
            {
              v58 = *(__int64 (**)())(v43 + 168);
              v59 = -1;
              if ( v58 != sub_2E77FD0 )
              {
                v82 = v40;
                v59 = ((__int64 (__fastcall *)(__int64 *, __int64))v58)(v42, v39);
                v40 = v82;
              }
            }
            else
            {
              v80 = v40;
              v59 = ((__int64 (__fastcall *)(__int64 *, __int64))v57)(v42, v39);
              v40 = v80;
            }
            v31 += v59;
          }
          else
          {
            v44 = *(__int64 (**)())(v43 + 168);
            v45 = -1;
            if ( v44 != sub_2E77FD0 )
            {
              v79 = v40;
              v45 = ((__int64 (__fastcall *)(__int64 *, __int64))v44)(v42, v39);
              v40 = v79;
            }
            v18 += v45;
          }
          if ( (*(_BYTE *)v39 & 4) == 0 )
            break;
          v39 = *(_QWORD *)(v39 + 8);
          if ( v40 == v39 )
            goto LABEL_66;
        }
        while ( (*(_BYTE *)(v39 + 44) & 8) != 0 )
          v39 = *(_QWORD *)(v39 + 8);
        v39 = *(_QWORD *)(v39 + 8);
      }
      while ( v40 != v39 );
    }
LABEL_66:
    v50 = a4 + 48;
    v51 = sub_2E313E0(a4);
    if ( a4 + 48 != v51 )
    {
      do
      {
        while ( 1 )
        {
          v56 = *(_DWORD *)(v51 + 44);
          if ( (v56 & 4) != 0 || (v56 & 8) == 0 )
            v52 = (*(_QWORD *)(*(_QWORD *)(v51 + 16) + 24LL) >> 10) & 1LL;
          else
            LOBYTE(v52) = sub_2E88A90(v51, 1024, 1);
          if ( (_BYTE)v52 )
          {
            v53 = *(_QWORD *)(a1 + 528);
            v54 = *(__int64 (**)())(*(_QWORD *)v53 + 448LL);
            if ( v54 != sub_2FDCF50 || (v54 = *(__int64 (**)())(*(_QWORD *)v53 + 168LL), v55 = -1, v54 != sub_2E77FD0) )
              v55 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v54)(v53, v51);
            v31 += v55;
          }
          if ( (*(_BYTE *)v51 & 4) == 0 )
            break;
          v51 = *(_QWORD *)(v51 + 8);
          if ( v50 == v51 )
            goto LABEL_85;
        }
        while ( (*(_BYTE *)(v51 + 44) & 8) != 0 )
          v51 = *(_QWORD *)(v51 + 8);
        v51 = *(_QWORD *)(v51 + 8);
      }
      while ( v50 != v51 );
    }
LABEL_85:
    v60 = v86;
    v61 = 0;
    v62 = v87;
    if ( v86 != v88 )
    {
      do
      {
        v61 -= ((unsigned __int16)(*(_WORD *)(v60 + 68) - 14) < 5u) - 1;
        if ( (*(_BYTE *)v60 & 4) == 0 && (*(_BYTE *)(v60 + 44) & 8) != 0 )
        {
          do
            v60 = *(_QWORD *)(v60 + 8);
          while ( (*(_BYTE *)(v60 + 44) & 8) != 0 );
        }
        v60 = *(_QWORD *)(v60 + 8);
      }
      while ( v88 != v60 );
      goto LABEL_96;
    }
    if ( v87 == v89[0] )
      goto LABEL_100;
    do
    {
      v61 -= ((unsigned __int16)(*(_WORD *)(v62 + 68) - 14) < 5u) - 1;
      if ( (*(_BYTE *)v62 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v62 + 44) & 8) != 0 )
          v62 = *(_QWORD *)(v62 + 8);
      }
      v62 = *(_QWORD *)(v62 + 8);
LABEL_96:
      ;
    }
    while ( v62 != v89[0] );
    result = 0;
    if ( v61 <= 0xF )
    {
LABEL_100:
      v63 = *(_QWORD *)(a1 + 528);
      v64 = *(__int64 (**)())(*(_QWORD *)v63 + 440LL);
      v65 = 0;
      if ( v64 != sub_2FDC560 )
        v65 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v64)(v63, v66, v61);
      return (v18 >> 1) + v31 > v65;
    }
  }
  else
  {
    result = *(_DWORD *)(a3 + 4) + *(_DWORD *)(a3 + 8) != a5 && *(_DWORD *)(a2 + 4) + *(_DWORD *)(a2 + 8) != a5;
    if ( result )
    {
      v12 = *(_QWORD *)(a1 + 528);
      v13 = *(__int64 (**)())(*(_QWORD *)v12 + 424LL);
      result = 0;
      if ( v13 != sub_2DB1B00 )
        return ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64))v13)(
                 v12,
                 *(_QWORD *)(a2 + 16),
                 (unsigned int)(*(_DWORD *)(a2 + 4) + *(_DWORD *)(a2 + 8) - a5),
                 *(unsigned int *)(a2 + 12),
                 *(_QWORD *)(a3 + 16),
                 (unsigned int)(*(_DWORD *)(a3 + 4) + *(_DWORD *)(a3 + 8) - a5),
                 *(unsigned int *)(a3 + 12),
                 v7);
    }
  }
  return result;
}
