// Function: sub_280AC30
// Address: 0x280ac30
//
__int64 __fastcall sub_280AC30(_QWORD *a1)
{
  _QWORD *v1; // rcx
  _QWORD *v2; // rbx
  __int64 v3; // rax
  _BYTE *v4; // r12
  int v5; // r10d
  unsigned int v6; // ecx
  _BYTE *v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  int *v10; // rax
  int *v11; // r15
  int *v12; // rax
  int *v13; // rdx
  unsigned __int64 v14; // rsi
  _BYTE *v15; // r9
  _BYTE *v16; // r11
  _BYTE *v17; // rcx
  _BYTE *v18; // rdx
  __int64 i; // r12
  __int64 v20; // rbx
  __int64 v21; // r13
  unsigned __int64 v22; // rbx
  _QWORD *v23; // r12
  unsigned __int64 v24; // r14
  __int64 v25; // rax
  _QWORD *v26; // r13
  _QWORD *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned int v34; // r12d
  int v36; // ecx
  __int64 v37; // r9
  _BYTE *v38; // r10
  __int64 v39; // r11
  __int64 v40; // rax
  _QWORD *v41; // rbx
  _QWORD *v42; // r15
  __int64 v43; // rsi
  unsigned int v44; // edx
  _BYTE *v45; // r8
  int v46; // r10d
  _QWORD *v47; // r9
  _QWORD *v48; // r8
  unsigned int v49; // r13d
  int v50; // r9d
  __int64 v51; // rsi
  _BYTE *v52; // r10
  __int64 v53; // r8
  _BYTE *v54; // rcx
  __int64 v55; // r8
  _BYTE *v56; // rdi
  _BYTE *v57; // rax
  _QWORD *v58; // r8
  _BYTE *v59; // rdi
  _BYTE *v60; // rax
  _QWORD *v61; // r8
  _QWORD *v63; // [rsp+10h] [rbp-110h]
  _QWORD *v64; // [rsp+18h] [rbp-108h]
  _QWORD *v65; // [rsp+20h] [rbp-100h]
  _QWORD *v66; // [rsp+38h] [rbp-E8h]
  __int64 v67; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v68; // [rsp+48h] [rbp-D8h]
  __int64 v69; // [rsp+50h] [rbp-D0h]
  unsigned int v70; // [rsp+58h] [rbp-C8h]
  void *v71; // [rsp+60h] [rbp-C0h]
  _QWORD v72[2]; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v73; // [rsp+78h] [rbp-A8h]
  __int64 v74; // [rsp+80h] [rbp-A0h]
  void *v75; // [rsp+90h] [rbp-90h] BYREF
  __int64 v76; // [rsp+98h] [rbp-88h] BYREF
  _QWORD *v77; // [rsp+A0h] [rbp-80h]
  __int64 v78; // [rsp+A8h] [rbp-78h]
  __int64 v79; // [rsp+B0h] [rbp-70h]
  __int64 v80; // [rsp+C0h] [rbp-60h] BYREF
  int v81; // [rsp+C8h] [rbp-58h] BYREF
  unsigned __int64 v82; // [rsp+D0h] [rbp-50h]
  int *v83; // [rsp+D8h] [rbp-48h]
  int *v84; // [rsp+E0h] [rbp-40h]
  __int64 v85; // [rsp+E8h] [rbp-38h]

  v1 = (_QWORD *)*a1;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v81 = 0;
  v82 = 0;
  v83 = &v81;
  v84 = &v81;
  v85 = 0;
  v64 = v1;
  if ( v1 == a1 )
  {
    v33 = 0;
    v34 = 0;
    goto LABEL_60;
  }
  do
  {
    v2 = (_QWORD *)v64[6];
    v3 = *((unsigned int *)v64 + 14);
    if ( v2 == &v2[v3] )
      goto LABEL_20;
    v66 = &v2[v3];
    do
    {
      while ( 1 )
      {
        v4 = (_BYTE *)*v2;
        if ( *(_BYTE *)*v2 == 61 )
          break;
LABEL_4:
        if ( v66 == ++v2 )
          goto LABEL_20;
      }
      if ( !v70 )
      {
        ++v67;
        goto LABEL_105;
      }
      v5 = 1;
      v6 = (v70 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v7 = *(_BYTE **)(v68 + 16LL * v6);
      v65 = (_QWORD *)(v68 + 16LL * v6);
      v8 = 0;
      if ( v4 != v7 )
      {
        while ( v7 != (_BYTE *)-4096LL )
        {
          if ( !v8 && v7 == (_BYTE *)-8192LL )
            v8 = v65;
          v6 = (v70 - 1) & (v5 + v6);
          v65 = (_QWORD *)(v68 + 16LL * v6);
          v7 = (_BYTE *)*v65;
          if ( v4 == (_BYTE *)*v65 )
            goto LABEL_8;
          ++v5;
        }
        if ( !v8 )
          v8 = v65;
        ++v67;
        v36 = v69 + 1;
        if ( 4 * ((int)v69 + 1) >= 3 * v70 )
        {
LABEL_105:
          sub_2808B80((__int64)&v67, 2 * v70);
          if ( !v70 )
            goto LABEL_149;
          v44 = (v70 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v36 = v69 + 1;
          v8 = (_QWORD *)(v68 + 16LL * v44);
          v45 = (_BYTE *)*v8;
          if ( v4 != (_BYTE *)*v8 )
          {
            v46 = 1;
            v47 = 0;
            while ( v45 != (_BYTE *)-4096LL )
            {
              if ( !v47 && v45 == (_BYTE *)-8192LL )
                v47 = v8;
              v44 = (v70 - 1) & (v46 + v44);
              v8 = (_QWORD *)(v68 + 16LL * v44);
              v45 = (_BYTE *)*v8;
              if ( v4 == (_BYTE *)*v8 )
                goto LABEL_71;
              ++v46;
            }
            if ( v47 )
              v8 = v47;
          }
        }
        else if ( v70 - HIDWORD(v69) - v36 <= v70 >> 3 )
        {
          sub_2808B80((__int64)&v67, v70);
          if ( !v70 )
          {
LABEL_149:
            LODWORD(v69) = v69 + 1;
            BUG();
          }
          v48 = 0;
          v49 = (v70 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v50 = 1;
          v36 = v69 + 1;
          v8 = (_QWORD *)(v68 + 16LL * v49);
          v51 = *v8;
          if ( v4 != (_BYTE *)*v8 )
          {
            while ( v51 != -4096 )
            {
              if ( !v48 && v51 == -8192 )
                v48 = v8;
              v49 = (v70 - 1) & (v50 + v49);
              v8 = (_QWORD *)(v68 + 16LL * v49);
              v51 = *v8;
              if ( v4 == (_BYTE *)*v8 )
                goto LABEL_71;
              ++v50;
            }
            if ( v48 )
              v8 = v48;
          }
        }
LABEL_71:
        LODWORD(v69) = v36;
        if ( *v8 != -4096 )
          --HIDWORD(v69);
        *v8 = v4;
        v8[1] = v64 + 2;
        goto LABEL_4;
      }
LABEL_8:
      v63 = v2;
      v9 = v64;
      do
      {
        v9 = (_QWORD *)v9[1];
        v75 = &v75;
        v76 = 1;
        v77 = v64 + 2;
        v10 = (int *)sub_2808560(&v80, (__int64)&v75);
        v77 = v9 + 2;
        v75 = &v75;
        v11 = v10;
        v76 = 1;
        v12 = (int *)sub_2808560(&v80, (__int64)&v75);
        v13 = v12;
        if ( v12 == &v81 )
        {
          if ( v11 == &v81 )
            continue;
          if ( (v11[10] & 1) != 0 )
          {
            *(_QWORD *)(*((_QWORD *)v11 + 4) + 8LL) &= 1uLL;
            BUG();
          }
          v18 = (_BYTE *)*((_QWORD *)v11 + 4);
          v14 = 0;
          if ( (v18[8] & 1) != 0 )
            goto LABEL_17;
          goto LABEL_77;
        }
        if ( (v12[10] & 1) != 0 )
        {
          v14 = (unsigned __int64)(v12 + 8);
LABEL_84:
          if ( v11 == &v81 )
            goto LABEL_148;
LABEL_85:
          v18 = v11 + 8;
          if ( (v11[10] & 1) != 0 )
            goto LABEL_87;
          v18 = (_BYTE *)*((_QWORD *)v11 + 4);
          if ( (v18[8] & 1) != 0 )
            goto LABEL_87;
LABEL_77:
          v37 = *(_QWORD *)v18;
          if ( (*(_BYTE *)(*(_QWORD *)v18 + 8LL) & 1) != 0 )
          {
            v18 = *(_BYTE **)v18;
          }
          else
          {
            v38 = *(_BYTE **)v37;
            if ( (*(_BYTE *)(*(_QWORD *)v37 + 8LL) & 1) == 0 )
            {
              v39 = *(_QWORD *)v38;
              if ( (*(_BYTE *)(*(_QWORD *)v38 + 8LL) & 1) != 0 )
              {
                v38 = *(_BYTE **)v38;
              }
              else
              {
                v54 = *(_BYTE **)v39;
                if ( (*(_BYTE *)(*(_QWORD *)v39 + 8LL) & 1) == 0 )
                {
                  v55 = *(_QWORD *)v54;
                  if ( (*(_BYTE *)(*(_QWORD *)v54 + 8LL) & 1) != 0 )
                  {
                    v54 = *(_BYTE **)v54;
                  }
                  else
                  {
                    v59 = *(_BYTE **)v55;
                    if ( (*(_BYTE *)(*(_QWORD *)v55 + 8LL) & 1) == 0 )
                    {
                      v60 = sub_2808AF0(v59);
                      *v61 = v60;
                      v59 = v60;
                    }
                    *(_QWORD *)v54 = v59;
                    v54 = v59;
                  }
                  *(_QWORD *)v39 = v54;
                }
                *(_QWORD *)v38 = v54;
                v38 = v54;
              }
              *(_QWORD *)v37 = v38;
            }
            *(_QWORD *)v18 = v38;
            v18 = v38;
          }
          *((_QWORD *)v11 + 4) = v18;
          goto LABEL_87;
        }
        v14 = *((_QWORD *)v12 + 4);
        if ( (*(_BYTE *)(v14 + 8) & 1) != 0 )
          goto LABEL_84;
        v15 = *(_BYTE **)v14;
        if ( (*(_BYTE *)(*(_QWORD *)v14 + 8LL) & 1) != 0 )
        {
          *((_QWORD *)v12 + 4) = v15;
          if ( v11 == &v81 )
LABEL_148:
            BUG();
          v14 = (unsigned __int64)v15;
          goto LABEL_85;
        }
        v16 = *(_BYTE **)v15;
        if ( (*(_BYTE *)(*(_QWORD *)v15 + 8LL) & 1) != 0 )
        {
          *(_QWORD *)v14 = v16;
          v14 = (unsigned __int64)v16;
          *((_QWORD *)v12 + 4) = v16;
          if ( v11 == &v81 )
          {
            v18 = 0;
            goto LABEL_17;
          }
          goto LABEL_85;
        }
        v17 = *(_BYTE **)v16;
        if ( (*(_BYTE *)(*(_QWORD *)v16 + 8LL) & 1) != 0 )
        {
          *(_QWORD *)v15 = v17;
          *(_QWORD *)v14 = v17;
          v14 = (unsigned __int64)v17;
          *((_QWORD *)v12 + 4) = v17;
          if ( v11 == &v81 )
          {
            v18 = 0;
LABEL_17:
            *(_QWORD *)(*(_QWORD *)v18 + 8LL) = v14 | *(_QWORD *)(*(_QWORD *)v18 + 8LL) & 1LL;
            *(_QWORD *)v18 = *(_QWORD *)v14;
            *(_QWORD *)(v14 + 8) &= ~1uLL;
            *(_QWORD *)v14 = v18;
            continue;
          }
          goto LABEL_85;
        }
        v52 = *(_BYTE **)v17;
        if ( (*(_BYTE *)(*(_QWORD *)v17 + 8LL) & 1) == 0 )
        {
          v53 = *(_QWORD *)v52;
          if ( (*(_BYTE *)(*(_QWORD *)v52 + 8LL) & 1) != 0 )
          {
            v52 = *(_BYTE **)v52;
          }
          else
          {
            v56 = *(_BYTE **)v53;
            if ( (*(_BYTE *)(*(_QWORD *)v53 + 8LL) & 1) == 0 )
            {
              v57 = sub_2808AF0(v56);
              *v58 = v57;
              v56 = v57;
            }
            *(_QWORD *)v52 = v56;
            v52 = v56;
          }
          *(_QWORD *)v17 = v52;
        }
        *(_QWORD *)v16 = v52;
        *(_QWORD *)v15 = v52;
        *(_QWORD *)v14 = v52;
        v14 = (unsigned __int64)v52;
        *((_QWORD *)v13 + 4) = v52;
        v18 = 0;
        if ( v11 != &v81 )
          goto LABEL_85;
LABEL_87:
        if ( (_BYTE *)v14 != v18 )
          goto LABEL_17;
      }
      while ( (_QWORD *)v65[1] != v9 + 2 );
      v2 = v63 + 1;
    }
    while ( v66 != v63 + 1 );
LABEL_20:
    v64 = (_QWORD *)*v64;
  }
  while ( a1 != v64 );
  if ( v85 )
  {
    for ( i = (__int64)v83; (int *)i != &v81; i = sub_220EF30(i) )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(i + 40);
        if ( (v20 & 1) != 0 )
        {
          v21 = *(_QWORD *)(i + 48);
          v22 = v20 & 0xFFFFFFFFFFFFFFFELL;
          if ( v22 )
            break;
        }
        i = sub_220EF30(i);
        if ( (int *)i == &v81 )
          goto LABEL_29;
      }
      do
      {
        sub_280A780(*(_QWORD *)(v22 + 16), v21);
        v22 = *(_QWORD *)(v22 + 8) & 0xFFFFFFFFFFFFFFFELL;
      }
      while ( v22 );
    }
LABEL_29:
    v23 = (_QWORD *)*a1;
    while ( v64 != v23 )
    {
      while ( 1 )
      {
        v24 = (unsigned __int64)v23;
        v23 = (_QWORD *)*v23;
        if ( !*(_DWORD *)(v24 + 56) )
          break;
        if ( v64 == v23 )
          goto LABEL_59;
      }
      --a1[2];
      sub_2208CA0((__int64 *)v24);
      if ( *(_BYTE *)(v24 + 296) )
      {
        v40 = *(unsigned int *)(v24 + 288);
        *(_BYTE *)(v24 + 296) = 0;
        if ( (_DWORD)v40 )
        {
          v41 = *(_QWORD **)(v24 + 272);
          v42 = &v41[2 * v40];
          do
          {
            if ( *v41 != -8192 && *v41 != -4096 )
            {
              v43 = v41[1];
              if ( v43 )
                sub_B91220((__int64)(v41 + 1), v43);
            }
            v41 += 2;
          }
          while ( v42 != v41 );
          LODWORD(v40) = *(_DWORD *)(v24 + 288);
        }
        sub_C7D6A0(*(_QWORD *)(v24 + 272), 16LL * (unsigned int)v40, 8);
      }
      v25 = *(unsigned int *)(v24 + 256);
      if ( (_DWORD)v25 )
      {
        v26 = *(_QWORD **)(v24 + 240);
        v72[0] = 2;
        v72[1] = 0;
        v27 = &v26[8 * v25];
        v73 = -4096;
        v28 = -4096;
        v71 = &unk_49DD7B0;
        v75 = &unk_49DD7B0;
        v74 = 0;
        v76 = 2;
        v77 = 0;
        v78 = -8192;
        v79 = 0;
        while ( 1 )
        {
          v29 = v26[3];
          if ( v28 != v29 )
          {
            v28 = v78;
            if ( v29 != v78 )
            {
              v30 = v26[7];
              if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
              {
                sub_BD60C0(v26 + 5);
                v29 = v26[3];
              }
              v28 = v29;
            }
          }
          *v26 = &unk_49DB368;
          if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
            sub_BD60C0(v26 + 1);
          v26 += 8;
          if ( v27 == v26 )
            break;
          v28 = v73;
        }
        v75 = &unk_49DB368;
        if ( v78 != 0 && v78 != -4096 && v78 != -8192 )
          sub_BD60C0(&v76);
        v71 = &unk_49DB368;
        if ( v73 != 0 && v73 != -4096 && v73 != -8192 )
          sub_BD60C0(v72);
        LODWORD(v25) = *(_DWORD *)(v24 + 256);
      }
      sub_C7D6A0(*(_QWORD *)(v24 + 240), (unsigned __int64)(unsigned int)v25 << 6, 8);
      v31 = *(_QWORD *)(v24 + 152);
      if ( v31 != v24 + 168 )
        _libc_free(v31);
      v32 = *(_QWORD *)(v24 + 48);
      if ( v32 != v24 + 64 )
        _libc_free(v32);
      sub_C7D6A0(*(_QWORD *)(v24 + 24), 8LL * *(unsigned int *)(v24 + 40), 8);
      j_j___libc_free_0(v24);
    }
LABEL_59:
    v33 = v82;
    v34 = 1;
  }
  else
  {
    v33 = v82;
    v34 = 0;
  }
LABEL_60:
  sub_2808920(v33);
  sub_C7D6A0(v68, 16LL * v70, 8);
  return v34;
}
