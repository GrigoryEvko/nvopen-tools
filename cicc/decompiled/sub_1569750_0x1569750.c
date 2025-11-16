// Function: sub_1569750
// Address: 0x1569750
//
__int64 __fastcall sub_1569750(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r15
  int v6; // r12d
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rbx
  _BYTE *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  _BYTE *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  char v34; // si
  char v35; // si
  __int64 v36; // rax
  _QWORD *v37; // rcx
  _QWORD *v38; // r12
  unsigned __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rdx
  _QWORD *v42; // rsi
  _BYTE *v43; // rsi
  __int64 v44; // r8
  _QWORD *v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  unsigned int v51; // [rsp+Ch] [rbp-104h]
  __int64 v52; // [rsp+20h] [rbp-F0h]
  int v53; // [rsp+28h] [rbp-E8h]
  int v54; // [rsp+38h] [rbp-D8h]
  char v55; // [rsp+3Eh] [rbp-D2h]
  char v56; // [rsp+3Fh] [rbp-D1h]
  unsigned __int8 v58; // [rsp+48h] [rbp-C8h]
  _QWORD *v59; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+58h] [rbp-B8h]
  _QWORD v61[2]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD *v62; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v63; // [rsp+78h] [rbp-98h]
  _QWORD v64[2]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v65; // [rsp+90h] [rbp-80h] BYREF
  __int64 v66; // [rsp+98h] [rbp-78h]
  _QWORD v67[14]; // [rsp+A0h] [rbp-70h] BYREF

  sub_1C3DFC0(a1);
  v2 = sub_16327A0(a1);
  if ( !v2 )
    return 0;
  v5 = v2;
  v6 = sub_161F520(v2, a2, v3, v4);
  if ( !v6 )
    return 0;
  v58 = 0;
  v7 = 0;
  v56 = 0;
  v55 = 0;
  do
  {
    v8 = sub_161F530(v5, v7);
    v9 = v8;
    if ( *(_DWORD *)(v8 + 8) != 3 )
      goto LABEL_12;
    v10 = *(_BYTE **)(v8 - 16);
    if ( !v10 || *v10 )
      goto LABEL_12;
    v11 = sub_161E970(*(_QWORD *)(v8 - 16));
    if ( v12 == 30
      && !(*(_QWORD *)v11 ^ 0x76697463656A624FLL | *(_QWORD *)(v11 + 8) ^ 0x67616D4920432D65LL)
      && *(_QWORD *)(v11 + 16) == 0x56206F666E492065LL
      && *(_DWORD *)(v11 + 24) == 1769173605 )
    {
      v35 = v55;
      if ( *(_WORD *)(v11 + 28) == 28271 )
        v35 = 1;
      v55 = v35;
    }
    v13 = sub_161E970(v10);
    if ( v14 == 28
      && !(*(_QWORD *)v13 ^ 0x76697463656A624FLL | *(_QWORD *)(v13 + 8) ^ 0x73616C4320432D65LL)
      && *(_QWORD *)(v13 + 16) == 0x7265706F72502073LL )
    {
      v34 = v56;
      if ( *(_DWORD *)(v13 + 24) == 1936025972 )
        v34 = 1;
      v56 = v34;
    }
    v15 = sub_161E970(v10);
    if ( v16 == 9 && *(_QWORD *)v15 == 0x6576654C20434950LL && *(_BYTE *)(v15 + 8) == 108
      || (v17 = sub_161E970(v10), v18 == 9) && *(_QWORD *)v17 == 0x6576654C20454950LL && *(_BYTE *)(v17 + 8) == 108 )
    {
      v24 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      if ( v24 )
      {
        if ( *(_BYTE *)v24 == 1 )
        {
          v25 = *(_QWORD *)(v24 + 136);
          if ( *(_BYTE *)(v25 + 16) == 13 )
          {
            if ( *(_DWORD *)(v25 + 32) <= 0x40u )
            {
              v26 = *(_QWORD *)(v25 + 24);
              goto LABEL_32;
            }
            v52 = *(_QWORD *)(v24 + 136);
            v53 = *(_DWORD *)(v25 + 32);
            if ( v53 - (unsigned int)sub_16A57B0(v25 + 24) <= 0x40 )
            {
              v26 = **(_QWORD **)(v52 + 24);
LABEL_32:
              if ( v26 == 1 )
              {
                v27 = sub_1643350(*a1);
                v28 = sub_15A0680(v27, 7, 0);
                v65 = (_QWORD *)sub_1624210(v28, 7, v29, v30);
                v31 = sub_161E970(v10);
                v66 = sub_161FF10(*a1, v31, v32);
                v67[0] = *(_QWORD *)(v9 + 8 * (2LL - *(unsigned int *)(v9 + 8)));
                v33 = sub_1627350(*a1, &v65, 3, 0, 1);
                sub_1623BA0(v5, v7, v33);
                v58 = 1;
              }
            }
          }
        }
      }
    }
    v19 = sub_161E970(v10);
    if ( v20 == 30
      && !(*(_QWORD *)v19 ^ 0x76697463656A624FLL | *(_QWORD *)(v19 + 8) ^ 0x67616D4920432D65LL)
      && *(_QWORD *)(v19 + 16) == 0x53206F666E492065LL
      && *(_DWORD *)(v19 + 24) == 1769235301
      && *(_WORD *)(v19 + 28) == 28271 )
    {
      v22 = *(_BYTE **)(v9 + 8 * (2LL - *(unsigned int *)(v9 + 8)));
      if ( v22 )
      {
        if ( !*v22 )
        {
          v65 = v67;
          v66 = 0x400000000LL;
          v62 = (_QWORD *)sub_161E970(v22);
          v63 = v23;
          sub_16D2730(&v62, &v65, " ", 1, 0xFFFFFFFFLL, 1);
          if ( (unsigned int)v66 != 1 )
          {
            v36 = 2LL * (unsigned int)v66;
            v60 = 0;
            v37 = &v65[v36];
            v59 = v61;
            LOBYTE(v61[0]) = 0;
            if ( &v65[v36] == v65 )
            {
              v45 = v61;
              v44 = 0;
            }
            else
            {
              v54 = v6;
              v38 = &v65[v36];
              v51 = v7;
              v39 = (unsigned __int64)v65;
              do
              {
                v43 = *(_BYTE **)v39;
                if ( *(_QWORD *)v39 )
                {
                  v40 = *(_QWORD *)(v39 + 8);
                  v62 = v64;
                  sub_1564140((__int64 *)&v62, v43, (__int64)&v43[v40]);
                  v41 = v63;
                  v42 = v62;
                }
                else
                {
                  v42 = v64;
                  v62 = v64;
                  v41 = 0;
                  v63 = 0;
                  LOBYTE(v64[0]) = 0;
                }
                sub_2241490(&v59, v42, v41, v37);
                if ( v62 != v64 )
                  j_j___libc_free_0(v62, v64[0] + 1LL);
                v39 += 16LL;
              }
              while ( v38 != (_QWORD *)v39 );
              v6 = v54;
              v7 = v51;
              v44 = v60;
              v45 = v59;
            }
            v46 = *(unsigned int *)(v9 + 8);
            v62 = *(_QWORD **)(v9 - 8 * v46);
            v47 = *a1;
            v63 = *(_QWORD *)(v9 + 8 * (1 - v46));
            v48 = sub_161FF10(v47, v45, v44);
            v49 = *a1;
            v64[0] = v48;
            v50 = sub_1627350(v49, &v62, 3, 0, 1);
            sub_1623BA0(v5, v7, v50);
            if ( v59 != v61 )
              j_j___libc_free_0(v59, v61[0] + 1LL);
            v58 = 1;
          }
          if ( v65 != v67 )
            _libc_free((unsigned __int64)v65);
        }
      }
    }
LABEL_12:
    ++v7;
  }
  while ( v6 != v7 );
  if ( (((unsigned __int8)v56 ^ 1) & (unsigned __int8)v55) != 0 )
  {
    sub_1632AD0(a1, 4, "Objective-C Class Properties", 28, 0);
    return ((unsigned __int8)v56 ^ 1u) & (unsigned __int8)v55;
  }
  return v58;
}
