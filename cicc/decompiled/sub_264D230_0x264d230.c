// Function: sub_264D230
// Address: 0x264d230
//
__int64 __fastcall sub_264D230(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 **v5; // rbx
  __int64 *v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // rcx
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  volatile signed __int32 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 *v16; // r12
  __int64 (__fastcall *v17)(__int64); // rax
  unsigned __int64 *v18; // rdi
  unsigned __int64 *v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  _DWORD *v23; // rax
  unsigned __int64 *v24; // rbx
  __int64 (__fastcall *v25)(__int64); // rdx
  unsigned __int64 *v26; // r12
  unsigned __int64 *v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rbx
  unsigned __int64 v30; // r12
  volatile signed __int32 *v31; // rdi
  _DWORD *v33; // r12
  _DWORD *v34; // rbx
  unsigned int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // edx
  _DWORD *v38; // rdi
  int v39; // ecx
  _DWORD *v40; // rax
  int v41; // r11d
  _DWORD *v42; // r10
  int v43; // eax
  int v44; // edx
  int v45; // eax
  int v46; // eax
  __int64 v47; // r9
  unsigned int v48; // ecx
  int v49; // r15d
  int v50; // edi
  _DWORD *v51; // rsi
  int v52; // eax
  int v53; // eax
  __int64 v54; // r9
  int v55; // edi
  unsigned int v56; // ecx
  int v57; // r15d
  __int64 v58; // [rsp+8h] [rbp-C8h]
  __int64 v59; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v60; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+28h] [rbp-A8h]
  __int64 *v62; // [rsp+38h] [rbp-98h]
  unsigned __int64 v63; // [rsp+40h] [rbp-90h] BYREF
  __int64 v64; // [rsp+48h] [rbp-88h]
  __int64 v65; // [rsp+50h] [rbp-80h]
  __int64 v66; // [rsp+58h] [rbp-78h]
  _BYTE v67[16]; // [rsp+60h] [rbp-70h] BYREF
  __int64 (__fastcall *v68)(_QWORD *); // [rsp+70h] [rbp-60h]
  __int64 v69; // [rsp+78h] [rbp-58h]
  unsigned __int64 v70; // [rsp+80h] [rbp-50h] BYREF
  __int64 v71; // [rsp+88h] [rbp-48h]
  __int64 (__fastcall *v72)(__int64 *); // [rsp+90h] [rbp-40h]
  __int64 v73; // [rsp+98h] [rbp-38h]
  char v74; // [rsp+A0h] [rbp-30h] BYREF

  v5 = (__int64 **)(a2 + 48);
  v6 = *(__int64 **)(a2 + 56);
  v7 = *v5;
  if ( *v5 == v6 )
  {
    v7 = *(__int64 **)(a2 + 72);
    v6 = *(__int64 **)(a2 + 80);
    if ( v6 == v7 )
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
LABEL_81:
      *(_QWORD *)a1 = 1;
      goto LABEL_7;
    }
  }
  LODWORD(a3) = 0;
  do
  {
    v8 = *v7;
    v7 += 2;
    a3 = (unsigned int)(*(_DWORD *)(v8 + 40) + a3);
  }
  while ( v7 != v6 );
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( !(_DWORD)a3 )
    goto LABEL_81;
  v9 = sub_AF1560(4 * (int)a3 / 3u + 1);
  *(_QWORD *)a1 = 1;
  if ( v9 )
    sub_A08C50(a1, v9);
LABEL_7:
  if ( *(_BYTE *)a2 || byte_4FF3628 )
  {
    sub_2640E50(&v70, (_QWORD *)(a2 + 72), a3);
    sub_2640E50(&v60, &v70, v10);
  }
  else
  {
    v70 = 0;
    v71 = 0;
    v72 = 0;
    sub_2640E50(&v60, &v70, a3);
  }
  v62 = (__int64 *)v5;
  v11 = v70;
  v12 = v71;
  if ( v71 != v70 )
  {
    do
    {
      v13 = *(volatile signed __int32 **)(v11 + 8);
      if ( v13 )
        sub_A191D0(v13);
      v11 += 16LL;
    }
    while ( v12 != v11 );
    v11 = v70;
  }
  if ( v11 )
    j_j___libc_free_0(v11);
  v14 = *v62;
  v63 = v60;
  v58 = v61;
  v64 = v14;
  v15 = v62[1];
  v65 = v61;
  v66 = v15;
  v59 = v62[1];
  if ( v14 == v59 )
    goto LABEL_32;
  do
  {
    do
    {
      v16 = &v70;
      v72 = sub_263DA90;
      v17 = sub_263DA70;
      v18 = &v63;
      v73 = 0;
      v19 = &v70;
      if ( ((unsigned __int8)sub_263DA70 & 1) != 0 )
LABEL_19:
        v17 = *(__int64 (__fastcall **)(__int64))((char *)v17 + *v18 - 1);
      v20 = v17((__int64)v18);
      if ( !v20 )
      {
        while ( 1 )
        {
          v16 += 2;
          if ( v16 == (unsigned __int64 *)&v74 )
            break;
          v21 = v19[3];
          v17 = (__int64 (__fastcall *)(__int64))v19[2];
          v19 = v16;
          v18 = (unsigned __int64 *)((char *)&v63 + v21);
          if ( ((unsigned __int8)v17 & 1) != 0 )
            goto LABEL_19;
          v20 = v17((__int64)v18);
          if ( v20 )
            goto LABEL_24;
        }
LABEL_94:
        BUG();
      }
LABEL_24:
      v22 = *(_QWORD *)v20;
      v23 = *(_DWORD **)(*(_QWORD *)v20 + 32LL);
      if ( *(_DWORD *)(v22 + 40) )
      {
        v33 = &v23[*(unsigned int *)(v22 + 48)];
        if ( v23 != v33 )
        {
          while ( 1 )
          {
            v34 = v23;
            if ( *v23 <= 0xFFFFFFFD )
              break;
            if ( v33 == ++v23 )
              goto LABEL_25;
          }
          while ( v33 != v34 )
          {
            v35 = *(_DWORD *)(a1 + 24);
            if ( v35 )
            {
              v36 = *(_QWORD *)(a1 + 8);
              v37 = (v35 - 1) & (37 * *v34);
              v38 = (_DWORD *)(v36 + 4LL * v37);
              v39 = *v38;
              if ( *v34 == *v38 )
                goto LABEL_50;
              v41 = 1;
              v42 = 0;
              while ( v39 != -1 )
              {
                if ( v42 || v39 != -2 )
                  v38 = v42;
                v37 = (v35 - 1) & (v41 + v37);
                v39 = *(_DWORD *)(v36 + 4LL * v37);
                if ( *v34 == v39 )
                  goto LABEL_50;
                ++v41;
                v42 = v38;
                v38 = (_DWORD *)(v36 + 4LL * v37);
              }
              v43 = *(_DWORD *)(a1 + 16);
              if ( !v42 )
                v42 = v38;
              ++*(_QWORD *)a1;
              v44 = v43 + 1;
              if ( 4 * (v43 + 1) < 3 * v35 )
              {
                if ( v35 - *(_DWORD *)(a1 + 20) - v44 > v35 >> 3 )
                  goto LABEL_60;
                sub_A08C50(a1, v35);
                v52 = *(_DWORD *)(a1 + 24);
                if ( !v52 )
                {
LABEL_93:
                  ++*(_DWORD *)(a1 + 16);
                  BUG();
                }
                v53 = v52 - 1;
                v54 = *(_QWORD *)(a1 + 8);
                v55 = 1;
                v56 = v53 & (37 * *v34);
                v42 = (_DWORD *)(v54 + 4LL * v56);
                v44 = *(_DWORD *)(a1 + 16) + 1;
                v51 = 0;
                v57 = *v42;
                if ( *v42 == *v34 )
                  goto LABEL_60;
                while ( v57 != -1 )
                {
                  if ( !v51 && v57 == -2 )
                    v51 = v42;
                  v56 = v53 & (v56 + v55);
                  v42 = (_DWORD *)(v54 + 4LL * v56);
                  v57 = *v42;
                  if ( *v34 == *v42 )
                    goto LABEL_60;
                  ++v55;
                }
                goto LABEL_68;
              }
            }
            else
            {
              ++*(_QWORD *)a1;
            }
            sub_A08C50(a1, 2 * v35);
            v45 = *(_DWORD *)(a1 + 24);
            if ( !v45 )
              goto LABEL_93;
            v46 = v45 - 1;
            v47 = *(_QWORD *)(a1 + 8);
            v48 = v46 & (37 * *v34);
            v42 = (_DWORD *)(v47 + 4LL * v48);
            v44 = *(_DWORD *)(a1 + 16) + 1;
            v49 = *v42;
            if ( *v42 == *v34 )
              goto LABEL_60;
            v50 = 1;
            v51 = 0;
            while ( v49 != -1 )
            {
              if ( !v51 && v49 == -2 )
                v51 = v42;
              v48 = v46 & (v48 + v50);
              v42 = (_DWORD *)(v47 + 4LL * v48);
              v49 = *v42;
              if ( *v34 == *v42 )
                goto LABEL_60;
              ++v50;
            }
LABEL_68:
            if ( v51 )
              v42 = v51;
LABEL_60:
            *(_DWORD *)(a1 + 16) = v44;
            if ( *v42 != -1 )
              --*(_DWORD *)(a1 + 20);
            *v42 = *v34;
LABEL_50:
            v40 = v34 + 1;
            if ( v33 == v34 + 1 )
              break;
            while ( 1 )
            {
              v34 = v40;
              if ( *v40 <= 0xFFFFFFFD )
                break;
              if ( v33 == ++v40 )
                goto LABEL_25;
            }
          }
        }
      }
LABEL_25:
      v24 = (unsigned __int64 *)v67;
      v69 = 0;
      v68 = sub_263DA40;
      v25 = sub_263DA10;
      v26 = (unsigned __int64 *)v67;
      v27 = &v63;
      if ( ((unsigned __int8)sub_263DA10 & 1) != 0 )
LABEL_26:
        v25 = *(__int64 (__fastcall **)(__int64))((char *)v25 + *v27 - 1);
      while ( !(unsigned __int8)v25((__int64)v27) )
      {
        v24 += 2;
        if ( &v70 == v24 )
          goto LABEL_94;
        v28 = v26[3];
        v25 = (__int64 (__fastcall *)(__int64))v26[2];
        v26 = v24;
        v27 = (unsigned __int64 *)((char *)&v63 + v28);
        if ( ((unsigned __int8)v25 & 1) != 0 )
          goto LABEL_26;
      }
      v14 = v64;
    }
    while ( v64 != v59 );
LABEL_32:
    ;
  }
  while ( v58 != v63 || v14 != v66 || v58 != v65 );
  v29 = v61;
  v30 = v60;
  if ( v61 != v60 )
  {
    do
    {
      v31 = *(volatile signed __int32 **)(v30 + 8);
      if ( v31 )
        sub_A191D0(v31);
      v30 += 16LL;
    }
    while ( v29 != v30 );
    v30 = v60;
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  return a1;
}
