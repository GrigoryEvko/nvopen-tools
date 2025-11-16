// Function: sub_20C24E0
// Address: 0x20c24e0
//
void __fastcall sub_20C24E0(__int64 *a1, _QWORD *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // r15
  _QWORD *v5; // r14
  unsigned __int64 v6; // rdi
  __int16 v7; // ax
  __int64 v8; // rdx
  _QWORD *v9; // rcx
  _WORD *v10; // rbx
  __int64 v11; // rax
  __int64 *v12; // r11
  _QWORD *v13; // r9
  __int64 v14; // rsi
  int v15; // r12d
  __int64 v16; // rdx
  unsigned int v17; // edi
  __int16 v18; // ax
  _WORD *v19; // rdi
  unsigned __int16 *v20; // r8
  unsigned __int16 v21; // r14
  int v22; // edi
  unsigned __int16 *v23; // r10
  unsigned __int16 *v24; // r13
  unsigned __int16 *v25; // rax
  int v26; // r8d
  __int64 v27; // rbx
  _QWORD *v28; // rax
  int v29; // edx
  int v30; // eax
  unsigned __int16 *v31; // rax
  unsigned __int16 v32; // cx
  _QWORD *v33; // rdx
  __int64 v34; // rdi
  unsigned int v35; // esi
  _WORD *v36; // rdi
  unsigned __int16 v37; // ax
  _WORD *v38; // rsi
  unsigned __int16 v39; // ax
  __int64 v40; // rbx
  __int64 v41; // r10
  __int64 v42; // rdi
  _QWORD *v43; // rax
  int v44; // edx
  unsigned __int16 *v45; // rax
  __int64 v46; // rax
  unsigned __int16 v47; // ax
  __int16 v48; // ax
  __int64 *v49; // [rsp+0h] [rbp-E0h]
  __int64 *v51; // [rsp+10h] [rbp-D0h]
  _WORD *v52; // [rsp+18h] [rbp-C8h]
  _WORD *v53; // [rsp+20h] [rbp-C0h]
  _QWORD *v55; // [rsp+30h] [rbp-B0h]
  bool v56; // [rsp+3Bh] [rbp-A5h]
  int v57; // [rsp+3Ch] [rbp-A4h]
  __int64 *v58; // [rsp+40h] [rbp-A0h]
  _QWORD *v59; // [rsp+48h] [rbp-98h]
  _QWORD *v60; // [rsp+48h] [rbp-98h]
  unsigned __int16 *v61; // [rsp+48h] [rbp-98h]
  unsigned __int64 v62[4]; // [rsp+50h] [rbp-90h] BYREF
  int v63; // [rsp+70h] [rbp-70h] BYREF
  _QWORD *v64; // [rsp+78h] [rbp-68h]
  char v65; // [rsp+80h] [rbp-60h]
  unsigned __int16 v66; // [rsp+88h] [rbp-58h]
  _WORD *v67; // [rsp+90h] [rbp-50h]
  int v68; // [rsp+98h] [rbp-48h]
  unsigned __int16 v69; // [rsp+A0h] [rbp-40h]
  __int64 v70; // [rsp+A8h] [rbp-38h]

  v2 = *(_DWORD *)(a1[4] + 16);
  v3 = sub_22077B0(152);
  v4 = v3;
  if ( v3 )
    sub_20C2240(v3, v2, (__int64)a2);
  a1[9] = v4;
  v5 = a2 + 3;
  v6 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 3 == (_QWORD *)v6 )
  {
    v56 = 0;
    goto LABEL_13;
  }
  if ( !v6 )
    BUG();
  v7 = *(_WORD *)(v6 + 46);
  v8 = *(_QWORD *)v6;
  if ( (*(_QWORD *)v6 & 4) == 0 )
  {
    if ( (v7 & 4) != 0 )
    {
      while ( 1 )
      {
        v6 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        v7 = *(_WORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 46);
        if ( (v7 & 4) == 0 )
          break;
        v8 = *(_QWORD *)v6;
      }
    }
LABEL_10:
    if ( (v7 & 8) != 0 )
    {
      v56 = sub_1E15D00(v6, 8u, 1);
      v4 = a1[9];
      goto LABEL_13;
    }
    goto LABEL_59;
  }
  if ( (v7 & 4) == 0 )
    goto LABEL_10;
LABEL_59:
  v56 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) & 8LL) != 0;
LABEL_13:
  v9 = a2 + 3;
  v49 = (__int64 *)a2[12];
  v51 = (__int64 *)a2[11];
  if ( v51 != v49 )
  {
    do
    {
      v59 = v9;
      v10 = *(_WORD **)(*v51 + 160);
      v52 = v10;
      v11 = sub_1DD77D0(*v51);
      v12 = a1;
      v9 = v59;
      v53 = (_WORD *)v11;
      if ( v10 != (_WORD *)v11 )
      {
        do
        {
          v13 = (_QWORD *)v12[4];
          if ( !v13 )
            BUG();
          v14 = v13[1];
          v15 = 0;
          v16 = v13[7];
          v17 = *(_DWORD *)(v14 + 24LL * (unsigned __int16)*v53 + 16);
          v18 = v17 & 0xF;
          v19 = (_WORD *)(v16 + 2LL * (v17 >> 4));
          v20 = v19 + 1;
          v21 = *v19 + v18 * *v53;
          v22 = 0;
LABEL_17:
          v23 = v20;
          v24 = v20;
          if ( v20 )
          {
            while ( 1 )
            {
              v25 = (unsigned __int16 *)(v13[6] + 4LL * v21);
              v26 = *v25;
              v15 = v25[1];
              if ( (_WORD)v26 )
                break;
LABEL_52:
              v47 = *v23;
              v20 = 0;
              ++v23;
              v21 += v47;
              if ( !v47 )
                goto LABEL_17;
              v24 = v23;
              if ( !v23 )
                goto LABEL_54;
            }
            while ( 1 )
            {
              v27 = v16 + 2LL * *(unsigned int *)(v14 + 24LL * (unsigned __int16)v26 + 8);
              if ( v27 )
                break;
              if ( !(_WORD)v15 )
              {
                v22 = v26;
                goto LABEL_52;
              }
              v26 = v15;
              v15 = 0;
            }
          }
          else
          {
LABEL_54:
            v26 = v22;
            v27 = 0;
          }
          while ( 1 )
          {
            v60 = v13;
            if ( !v24 )
              break;
            while ( 1 )
            {
              v55 = v9;
              v57 = v26;
              v58 = v12;
              sub_20C2470((_QWORD *)v12[9], (unsigned __int16)v26, 0);
              v26 = v57;
              v9 = v55;
              v12 = v58;
              v13 = v60;
              v28 = (_QWORD *)a2[4];
              if ( v28 == v55 )
              {
                v29 = 0;
              }
              else
              {
                v29 = 0;
                do
                {
                  v28 = (_QWORD *)v28[1];
                  ++v29;
                }
                while ( v28 != v55 );
              }
              *(_DWORD *)(*(_QWORD *)(v4 + 104) + 4LL * (unsigned __int16)v57) = v29;
              v27 += 2;
              *(_DWORD *)(*(_QWORD *)(v4 + 128) + 4LL * (unsigned __int16)v57) = -1;
              v30 = *(unsigned __int16 *)(v27 - 2);
              if ( !(_WORD)v30 )
                break;
              v26 = v30 + v57;
            }
            if ( (_WORD)v15 )
            {
              v46 = (unsigned __int16)v15;
              v26 = v15;
              v15 = 0;
              v27 = v60[7] + 2LL * *(unsigned int *)(v60[1] + 24 * v46 + 8);
            }
            else
            {
              v15 = *v24;
              v21 += v15;
              if ( (_WORD)v15 )
              {
                ++v24;
                v45 = (unsigned __int16 *)(v60[6] + 4LL * v21);
                v26 = *v45;
                v15 = v45[1];
                v27 = v60[7] + 2LL * *(unsigned int *)(v60[1] + 24LL * (unsigned __int16)v26 + 8);
              }
              else
              {
                v27 = 0;
                v24 = 0;
              }
            }
          }
          v53 += 4;
        }
        while ( v52 != v53 );
      }
      ++v51;
    }
    while ( v49 != v51 );
    v5 = v9;
  }
  sub_1E08750((__int64)v62, *(_QWORD *)(a1[1] + 56), a1[1]);
  v31 = (unsigned __int16 *)sub_1E6A620(*(_QWORD **)(a1[1] + 40));
  v32 = *v31;
  if ( *v31 )
  {
    v61 = v31;
    do
    {
      if ( v56 || (*(_QWORD *)(v62[0] + 8LL * (v32 >> 6)) & (1LL << v32)) != 0 )
      {
        v33 = (_QWORD *)a1[4];
        v63 = v32;
        if ( !v33 )
        {
          v64 = 0;
          v65 = 1;
          v66 = 0;
          v67 = 0;
          v68 = 0;
          v69 = 0;
          v70 = 0;
          BUG();
        }
        v65 = 1;
        v66 = 0;
        v67 = 0;
        v70 = 0;
        v34 = v33[7];
        v64 = v33 + 1;
        v68 = 0;
        v69 = 0;
        v35 = *(_DWORD *)(v33[1] + 24LL * v32 + 16);
        v36 = (_WORD *)(v34 + 2LL * (v35 >> 4));
        v37 = *v36 + v32 * (v35 & 0xF);
        v38 = v36 + 1;
        v66 = v37;
        v67 = v36 + 1;
        while ( v38 )
        {
          v68 = *(_DWORD *)(v33[6] + 4LL * v66);
          v39 = v68;
          if ( (_WORD)v68 )
          {
            while ( 1 )
            {
              v40 = v39;
              v41 = *(unsigned int *)(v33[1] + 24LL * v39 + 8);
              v42 = v33[7];
              v69 = v39;
              v70 = v42 + 2 * v41;
              if ( v70 )
                break;
              v39 = HIWORD(v68);
              v68 = HIWORD(v68);
              if ( !v39 )
                goto LABEL_61;
            }
            while ( 1 )
            {
              sub_20C2470((_QWORD *)a1[9], v39, 0);
              v43 = (_QWORD *)a2[4];
              if ( v43 == v5 )
              {
                v44 = 0;
              }
              else
              {
                v44 = 0;
                do
                {
                  v43 = (_QWORD *)v43[1];
                  ++v44;
                }
                while ( v43 != v5 );
              }
              *(_DWORD *)(4 * v40 + *(_QWORD *)(v4 + 104)) = v44;
              *(_DWORD *)(*(_QWORD *)(v4 + 128) + 4 * v40) = -1;
              sub_1E1D5E0((__int64)&v63);
              if ( !v67 )
                break;
              v40 = v69;
              v39 = v69;
            }
            break;
          }
LABEL_61:
          v67 = ++v38;
          v48 = *(v38 - 1);
          v66 += v48;
          if ( !v48 )
          {
            v67 = 0;
            break;
          }
        }
      }
      v32 = *++v61;
    }
    while ( *v61 );
  }
  _libc_free(v62[0]);
}
