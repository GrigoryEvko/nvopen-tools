// Function: sub_18A98F0
// Address: 0x18a98f0
//
__int64 __fastcall sub_18A98F0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // r14
  unsigned int v12; // eax
  int v13; // ecx
  __int64 v14; // rdx
  unsigned int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  _QWORD *v19; // r10
  __int64 v20; // rdi
  unsigned __int64 v21; // r13
  __int64 v22; // rbx
  __int64 v23; // rax
  size_t *v24; // r14
  size_t **v25; // r12
  size_t v26; // r15
  _QWORD *v27; // r8
  _BYTE *v28; // rdi
  _BYTE *v29; // rax
  char *v30; // r13
  __int64 *v31; // r14
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // r8
  __int64 v35; // rax
  int v36; // r15d
  __int64 v37; // rax
  unsigned int v38; // eax
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 i; // r14
  _BYTE *v42; // rsi
  __int64 v43; // rax
  char v44; // al
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 *v47; // rdx
  __int64 *v48; // rsi
  __int64 v49; // rcx
  __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // rbx
  unsigned __int64 v55; // rdi
  __int64 *v56; // rbx
  __int64 *v57; // rdi
  __int64 v58; // rax
  _QWORD *v59; // [rsp+0h] [rbp-C0h]
  __int64 v60; // [rsp+8h] [rbp-B8h]
  __int64 v61; // [rsp+10h] [rbp-B0h]
  __int64 v62; // [rsp+18h] [rbp-A8h]
  __int64 v63; // [rsp+20h] [rbp-A0h]
  _QWORD *v64; // [rsp+28h] [rbp-98h]
  __int64 v65; // [rsp+30h] [rbp-90h]
  __int64 v66; // [rsp+38h] [rbp-88h]
  unsigned __int64 v67; // [rsp+40h] [rbp-80h]
  _QWORD *v68; // [rsp+40h] [rbp-80h]
  __int64 v69; // [rsp+48h] [rbp-78h]
  unsigned __int64 v70; // [rsp+48h] [rbp-78h]
  _QWORD *v71; // [rsp+48h] [rbp-78h]
  __int64 v72; // [rsp+58h] [rbp-68h] BYREF
  _QWORD *v73; // [rsp+60h] [rbp-60h] BYREF
  __int128 v74; // [rsp+68h] [rbp-58h]
  char v75; // [rsp+80h] [rbp-40h]

  v5 = a3 + 48;
  v6 = (__int64)a1;
  v8 = sub_15C70A0(a3 + 48);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v8 )
  {
    v9 = v8;
    v10 = sub_15C70A0(v5);
    v11 = v10 ? sub_393D1F0(*(_QWORD *)(a2 + 1200), v10) : *(_QWORD *)(a2 + 1200);
    if ( v11 )
    {
      v12 = sub_393D1C0(v9);
      v13 = 0;
      v14 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      if ( *(_BYTE *)v14 == 19 )
      {
        v15 = *(_DWORD *)(v14 + 24);
        if ( (v15 & 1) == 0 )
        {
          v13 = (v15 >> 1) & 0x1F;
          if ( ((v15 >> 1) & 0x20) != 0 )
            v13 |= (v15 >> 2) & 0xFE0;
        }
      }
      v73 = (_QWORD *)__PAIR64__(v13, v12);
      v17 = sub_18A8380(v11 + 32, (unsigned int *)&v73);
      if ( v17 == v11 + 40 )
      {
        v35 = sub_2241E40(v11 + 32, &v73, v16, v17, v18);
        v75 |= 1u;
        LODWORD(v73) = 0;
        *(_QWORD *)&v74 = v35;
        *a4 = 0;
      }
      else
      {
        v75 &= ~1u;
        v73 = 0;
        *(_QWORD *)&v74 = 0;
        *((_QWORD *)&v74 + 1) = 0x1000000000LL;
        if ( *(_DWORD *)(v17 + 60) )
        {
          v69 = v17;
          sub_16D1890((__int64)&v73, *(_DWORD *)(v17 + 56));
          v19 = v73;
          v20 = *(_QWORD *)(v69 + 48);
          v64 = v73;
          v63 = v20;
          *(_QWORD *)((char *)&v74 + 4) = *(_QWORD *)(v69 + 60);
          if ( (_DWORD)v74 )
          {
            v62 = v9;
            v59 = a4;
            v21 = 0;
            v22 = 8LL * (unsigned int)v74 + 8;
            v66 = 8LL * (unsigned int)(v74 - 1);
            v23 = v20;
            v61 = v11;
            v65 = v69;
            v60 = v6;
            while ( 1 )
            {
              v24 = *(size_t **)(v23 + v21);
              v25 = (size_t **)&v19[v21 / 8];
              if ( v24 != (size_t *)-8LL )
              {
                if ( v24 )
                  break;
              }
              *v25 = v24;
LABEL_14:
              v22 += 4;
              if ( v66 == v21 )
              {
                v9 = v62;
                v11 = v61;
                v6 = v60;
                a4 = v59;
                goto LABEL_47;
              }
              v19 = v73;
              v21 += 8LL;
              v23 = *(_QWORD *)(v65 + 48);
            }
            v26 = *v24;
            v67 = *v24 + 17;
            v70 = *v24 + 1;
            v27 = (_QWORD *)malloc(v67);
            if ( !v27 )
            {
              if ( !v67 )
              {
                v58 = malloc(1u);
                v27 = 0;
                if ( v58 )
                {
                  v28 = (_BYTE *)(v58 + 16);
                  v27 = (_QWORD *)v58;
                  goto LABEL_20;
                }
              }
              v68 = v27;
              sub_16BD1C0("Allocation failed", 1u);
              v27 = v68;
            }
            v28 = v27 + 2;
            if ( v70 <= 1 )
            {
LABEL_13:
              v28[v26] = 0;
              *v27 = v26;
              v27[1] = v24[1];
              *v25 = v27;
              *(_DWORD *)((char *)v64 + v22) = *(_DWORD *)(v63 + v22);
              goto LABEL_14;
            }
LABEL_20:
            v71 = v27;
            v29 = memcpy(v28, v24 + 2, v26);
            v27 = v71;
            v28 = v29;
            goto LABEL_13;
          }
LABEL_47:
          v44 = v75;
          *a4 = 0;
          if ( (v44 & 1) == 0 && (_DWORD)v74 )
          {
            if ( *v73 != -8 && *v73 )
            {
              v47 = v73;
            }
            else
            {
              v45 = v73 + 1;
              do
              {
                do
                {
                  v46 = *v45;
                  v47 = v45++;
                }
                while ( v46 == -8 );
              }
              while ( !v46 );
            }
            v48 = &v73[(unsigned int)v74];
            if ( v48 != v47 )
            {
              v49 = 0;
              while ( 1 )
              {
                v49 += *(_QWORD *)(*v47 + 8);
                v50 = v47 + 1;
                *a4 = v49;
                v51 = v47[1];
                if ( !v51 || v51 == -8 )
                {
                  do
                  {
                    do
                    {
                      v52 = v50[1];
                      ++v50;
                    }
                    while ( v52 == -8 );
                  }
                  while ( !v52 );
                }
                if ( v50 == v48 )
                  break;
                v47 = v50;
              }
            }
          }
        }
        else
        {
          *a4 = 0;
        }
      }
      v36 = 0;
      v37 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      if ( *(_BYTE *)v37 == 19 )
      {
        v38 = *(_DWORD *)(v37 + 24);
        if ( (v38 & 1) == 0 )
        {
          v36 = (v38 >> 1) & 0x1F;
          if ( ((v38 >> 1) & 0x20) != 0 )
            v36 |= (v38 >> 2) & 0xFE0;
        }
      }
      HIDWORD(v72) = v36;
      LODWORD(v72) = sub_393D1C0(v9);
      v39 = sub_18A83F0(v11 + 80, (unsigned int *)&v72);
      if ( v39 != v11 + 88 && *(_QWORD *)(v39 + 80) )
      {
        v40 = *(_QWORD *)(v39 + 64);
        for ( i = v39 + 48; i != v40; v40 = sub_220EF30(v40) )
        {
          v43 = sub_18A58D0(v40 + 64);
          v72 = v40 + 64;
          *a4 += v43;
          v42 = *(_BYTE **)(v6 + 8);
          if ( v42 == *(_BYTE **)(v6 + 16) )
          {
            sub_18A9760(v6, v42, &v72);
          }
          else
          {
            if ( v42 )
            {
              *(_QWORD *)v42 = v40 + 64;
              v42 = *(_BYTE **)(v6 + 8);
            }
            *(_QWORD *)(v6 + 8) = v42 + 8;
          }
        }
        v30 = *(char **)(v6 + 8);
        v31 = *(__int64 **)v6;
        if ( *(char **)v6 != v30 )
        {
          _BitScanReverse64(&v32, (v30 - (char *)v31) >> 3);
          sub_18A5D70(*(__int64 **)v6, *(char **)(v6 + 8), 2LL * (int)(63 - (v32 ^ 0x3F)));
          if ( v30 - (char *)v31 > 128 )
          {
            v56 = v31 + 16;
            sub_18A7030((char *)v31, (char *)v31 + 128);
            if ( v30 != (char *)(v31 + 16) )
            {
              do
              {
                v57 = v56++;
                sub_18A6D30(v57);
              }
              while ( v30 != (char *)v56 );
            }
          }
          else
          {
            sub_18A7030((char *)v31, v30);
          }
        }
      }
      if ( (v75 & 1) == 0 )
      {
        v33 = (unsigned __int64)v73;
        if ( DWORD1(v74) && (_DWORD)v74 )
        {
          v53 = 8LL * (unsigned int)v74;
          v54 = 0;
          do
          {
            v55 = *(_QWORD *)(v33 + v54);
            if ( v55 != -8 && v55 )
            {
              _libc_free(v55);
              v33 = (unsigned __int64)v73;
            }
            v54 += 8;
          }
          while ( v54 != v53 );
        }
        _libc_free(v33);
      }
    }
  }
  return v6;
}
