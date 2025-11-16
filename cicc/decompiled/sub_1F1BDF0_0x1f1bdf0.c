// Function: sub_1F1BDF0
// Address: 0x1f1bdf0
//
void __fastcall sub_1F1BDF0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  int v4; // r15d
  unsigned __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // r14
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r12
  _QWORD *v13; // r10
  _QWORD *v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  _QWORD *v20; // r10
  unsigned __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // r11
  __int64 *v28; // rcx
  unsigned int v29; // esi
  __int64 v30; // rax
  __int64 v31; // r11
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdi
  int v37; // r12d
  char v38; // al
  __int64 v39; // rbx
  __int64 *v40; // rax
  int *v41; // r10
  unsigned int v42; // ebx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 i; // rcx
  unsigned int v46; // esi
  __int64 v47; // rdi
  unsigned int v48; // ecx
  __int64 *v49; // rax
  __int64 v50; // r11
  __int64 v51; // r9
  __int64 v52; // rsi
  _QWORD *v53; // rcx
  _QWORD *v54; // rdx
  int v55; // eax
  int v56; // r8d
  __int64 v57; // [rsp+8h] [rbp-D8h]
  _QWORD *v58; // [rsp+20h] [rbp-C0h]
  __int64 v59; // [rsp+28h] [rbp-B8h]
  __int64 v60; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v62; // [rsp+40h] [rbp-A0h]
  int v63; // [rsp+40h] [rbp-A0h]
  unsigned int v64; // [rsp+40h] [rbp-A0h]
  _QWORD *v65; // [rsp+48h] [rbp-98h]
  unsigned __int64 v66; // [rsp+48h] [rbp-98h]
  unsigned __int64 v67; // [rsp+48h] [rbp-98h]
  __int64 v68; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v69; // [rsp+58h] [rbp-88h] BYREF
  __int64 v70; // [rsp+60h] [rbp-80h]
  _BYTE v71[120]; // [rsp+68h] [rbp-78h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL) + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL));
  v5 = *(unsigned int *)(v3 + 408);
  v6 = v4 & 0x7FFFFFFF;
  v7 = v4 & 0x7FFFFFFF;
  v8 = 8 * v7;
  if ( (v4 & 0x7FFFFFFFu) < (unsigned int)v5 )
  {
    v59 = *(_QWORD *)(*(_QWORD *)(v3 + 400) + 8LL * v6);
    if ( v59 )
      goto LABEL_3;
  }
  v42 = v6 + 1;
  if ( (unsigned int)v5 < v6 + 1 )
  {
    v51 = v42;
    if ( v42 < v5 )
    {
      *(_DWORD *)(v3 + 408) = v42;
    }
    else if ( v42 > v5 )
    {
      if ( v42 > (unsigned __int64)*(unsigned int *)(v3 + 412) )
      {
        sub_16CD150(v3 + 400, (const void *)(v3 + 416), v42, 8, 8 * v4, v42);
        v5 = *(unsigned int *)(v3 + 408);
        v8 = 8LL * (v4 & 0x7FFFFFFF);
        v51 = v42;
      }
      v43 = *(_QWORD *)(v3 + 400);
      v52 = *(_QWORD *)(v3 + 416);
      v53 = (_QWORD *)(v43 + 8 * v51);
      v54 = (_QWORD *)(v43 + 8 * v5);
      if ( v53 != v54 )
      {
        do
          *v54++ = v52;
        while ( v53 != v54 );
        v43 = *(_QWORD *)(v3 + 400);
      }
      *(_DWORD *)(v3 + 408) = v42;
      goto LABEL_42;
    }
  }
  v43 = *(_QWORD *)(v3 + 400);
LABEL_42:
  *(_QWORD *)(v43 + v8) = sub_1DBA290(v4);
  v59 = *(_QWORD *)(*(_QWORD *)(v3 + 400) + 8 * v7);
  sub_1DBB110((_QWORD *)v3, v59);
LABEL_3:
  v69 = v71;
  v70 = 0x400000000LL;
  v68 = a1 + 200;
  v9 = *(unsigned int *)(a2 + 8);
  v60 = 8 * v9;
  if ( !(_DWORD)v9 )
    return;
  v10 = 0;
  do
  {
    v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + v10) + 8LL);
    if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    v12 = *(_QWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v13 = (_QWORD *)v12;
    v14 = *(_QWORD **)(*(_QWORD *)(v12 + 24) + 32LL);
    v58 = v14;
    if ( (_QWORD *)v12 == v14 )
    {
LABEL_15:
      v17 = (unsigned __int64)v13;
    }
    else
    {
      while ( 1 )
      {
        v15 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v15 )
          BUG();
        v16 = *(_QWORD *)v15;
        v17 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v15 & 4) == 0 && (*(_BYTE *)(v15 + 46) & 4) != 0 )
        {
          while ( 1 )
          {
            v18 = v16 & 0xFFFFFFFFFFFFFFF8LL;
            v17 = v18;
            if ( (*(_BYTE *)(v18 + 46) & 4) == 0 )
              break;
            v16 = *(_QWORD *)v18;
          }
        }
        if ( (unsigned __int16)(**(_WORD **)(v17 + 16) - 12) > 1u )
          break;
        v13 = (_QWORD *)v17;
        if ( (_QWORD *)v17 == v14 )
          goto LABEL_15;
      }
    }
    v62 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    v65 = v13;
    sub_1DBEA10(*(_QWORD *)(a1 + 16), v59, v11);
    sub_1F10740(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL), v12);
    sub_1E16240(v12);
    v20 = v65;
    v21 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    v22 = (v11 >> 1) & 3;
    LODWORD(v23) = v22;
    if ( ((v11 >> 1) & 3) != 0 )
      v24 = v62 | (2LL * ((int)v22 - 1));
    else
      v24 = *(_QWORD *)v62 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v25 = v68;
    v26 = *(unsigned int *)(v68 + 184);
    if ( (_DWORD)v26 )
    {
      sub_1F17CD0((__int64)&v68, v24, v26, v19, v22, v62);
      v33 = (unsigned int)v70;
      v20 = v65;
      v21 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = (v11 >> 1) & 3;
    }
    else
    {
      v27 = *(unsigned int *)(v68 + 188);
      if ( (_DWORD)v27 )
      {
        v28 = (__int64 *)(v68 + 8);
        v29 = *(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v24 >> 1) & 3;
        do
        {
          if ( (*(_DWORD *)((*v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v28 >> 1) & 3) > v29 )
            break;
          v26 = (unsigned int)(v26 + 1);
          v28 += 2;
        }
        while ( (_DWORD)v27 != (_DWORD)v26 );
      }
      LODWORD(v70) = 0;
      v30 = 0;
      v31 = (v26 << 32) | v27;
      if ( !HIDWORD(v70) )
      {
        v57 = v31;
        sub_16CD150((__int64)&v69, v71, 0, 16, v23, v62);
        v31 = v57;
        v23 = (v11 >> 1) & 3;
        v21 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        v20 = v65;
        v30 = 16LL * (unsigned int)v70;
      }
      v32 = (unsigned __int64)v69;
      *(_QWORD *)&v69[v30] = v25;
      *(_QWORD *)(v32 + v30 + 8) = v31;
      v33 = (unsigned int)(v70 + 1);
      LODWORD(v70) = v70 + 1;
    }
    if ( (_DWORD)v33 )
    {
      if ( *((_DWORD *)v69 + 3) < *((_DWORD *)v69 + 2) )
      {
        v34 = (unsigned __int64)&v69[16 * v33 - 16];
        v35 = *(_QWORD *)v34;
        v36 = 16LL * *(unsigned int *)(v34 + 12);
        if ( ((unsigned int)v23 | *(_DWORD *)(v21 + 24)) > (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)v34 + v36)
                                                                       & 0xFFFFFFFFFFFFFFF8LL)
                                                                      + 24)
                                                          | (unsigned int)(*(__int64 *)(*(_QWORD *)v34 + v36) >> 1) & 3)
          && *(_QWORD *)(v35 + v36 + 8) == v11 )
        {
          v37 = *(_DWORD *)(v35 + 4LL * *(unsigned int *)(v34 + 12) + 144);
          if ( v20 != v58 )
          {
            v63 = v23;
            v66 = v21;
            v38 = sub_1E166B0(v17, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL), 0);
            v21 = v66;
            LODWORD(v23) = v63;
            if ( v38 )
            {
              v44 = v17;
              for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
                    (*(_BYTE *)(v44 + 46) & 4) != 0;
                    v44 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL )
              {
                ;
              }
              v46 = *(_DWORD *)(i + 384);
              v47 = *(_QWORD *)(i + 368);
              if ( v46 )
              {
                v48 = (v46 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                v49 = (__int64 *)(v47 + 16LL * v48);
                v50 = *v49;
                if ( v44 == *v49 )
                {
LABEL_49:
                  sub_1F19510((__int64)&v68, v49[1] & 0xFFFFFFFFFFFFFFF8LL | 4);
                  goto LABEL_36;
                }
                v55 = 1;
                while ( v50 != -8 )
                {
                  v56 = v55 + 1;
                  v48 = (v46 - 1) & (v55 + v48);
                  v49 = (__int64 *)(v47 + 16LL * v48);
                  v50 = *v49;
                  if ( v44 == *v49 )
                    goto LABEL_49;
                  v55 = v56;
                }
              }
              v49 = (__int64 *)(v47 + 16LL * v46);
              goto LABEL_49;
            }
          }
          v64 = v23;
          v67 = v21;
          v39 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
          v40 = (__int64 *)sub_1DB3C70((__int64 *)v39, v11);
          v41 = 0;
          if ( v40 != (__int64 *)(*(_QWORD *)v39 + 24LL * *(unsigned int *)(v39 + 8))
            && (*(_DWORD *)((*v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v40 >> 1) & 3)) <= (*(_DWORD *)(v67 + 24) | v64) )
          {
            v41 = (int *)v40[2];
          }
          sub_1F1B3E0(a1, v37, v41);
        }
      }
    }
LABEL_36:
    v10 += 8;
  }
  while ( v10 != v60 );
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
}
