// Function: sub_31047D0
// Address: 0x31047d0
//
__int64 __fastcall sub_31047D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r15
  int v16; // ecx
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  _QWORD *v22; // rbx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // r12
  _QWORD *v26; // rbx
  __int64 v27; // rax
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // r13
  __int64 j; // rcx
  __int64 v31; // rax
  unsigned int v32; // r8d
  __int64 *v33; // r12
  __int64 v34; // r9
  unsigned int v35; // ecx
  __int64 *v36; // r13
  __int64 v37; // rdx
  __int64 v38; // rdx
  _QWORD *v39; // r12
  int v40; // r10d
  int v41; // edx
  _QWORD *v42; // rax
  unsigned int v43; // r10d
  __int64 v44; // rdi
  int v45; // esi
  __int64 *v46; // rcx
  int v47; // esi
  unsigned int v48; // r10d
  __int64 v49; // rdi
  int v50; // r9d
  __int64 v51; // rsi
  int i; // eax
  int v53; // r8d
  int v54; // r9d
  __int64 v55; // rsi
  __int64 v59; // [rsp+18h] [rbp-C8h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  __int64 v63; // [rsp+38h] [rbp-A8h]
  _QWORD *v64; // [rsp+40h] [rbp-A0h]
  unsigned int v65; // [rsp+48h] [rbp-98h]
  char v66; // [rsp+4Fh] [rbp-91h]
  __int64 (__fastcall **v67)(); // [rsp+50h] [rbp-90h] BYREF
  __int64 v68; // [rsp+58h] [rbp-88h] BYREF
  _QWORD *v69; // [rsp+60h] [rbp-80h]
  __int64 v70; // [rsp+68h] [rbp-78h]
  unsigned int v71; // [rsp+70h] [rbp-70h]
  _QWORD v72[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v73; // [rsp+90h] [rbp-50h]
  __int64 v74; // [rsp+98h] [rbp-48h]
  unsigned int v75; // [rsp+A0h] [rbp-40h]
  __int16 v76; // [rsp+A8h] [rbp-38h]

  v59 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = *(_QWORD *)(a3 + 80);
  v68 = 0;
  v69 = 0;
  v63 = v6 + 8;
  v67 = off_4A32990;
  v70 = 0;
  v71 = 0;
  v61 = a3 + 72;
  if ( a3 + 72 == v7 )
  {
    v62 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( *(_QWORD *)(v7 + 32) == v7 + 24 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( a3 + 72 == v7 )
        goto LABEL_7;
      if ( !v7 )
        BUG();
    }
    v62 = *(_QWORD *)(v7 + 32);
  }
  if ( v61 != v7 )
  {
    v13 = v7;
    while ( 1 )
    {
      if ( !v62 )
        BUG();
      v14 = *(_QWORD *)(v62 + 16);
      v15 = v62 - 24;
      v16 = *(_DWORD *)(v59 + 32);
      v17 = *(_QWORD *)(v59 + 16);
      if ( !v16 )
        goto LABEL_39;
      v18 = v16 - 1;
      v19 = v18 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v14 != *v20 )
      {
        for ( i = 1; ; i = v53 )
        {
          if ( v21 == -4096 )
            goto LABEL_39;
          v53 = i + 1;
          v19 = v18 & (i + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v14 == *v20 )
            break;
        }
      }
      v22 = (_QWORD *)v20[1];
      if ( v22 )
        break;
LABEL_39:
      for ( j = *(_QWORD *)(v62 + 8); ; j = *(_QWORD *)(v13 + 32) )
      {
        v31 = v13 - 24;
        if ( !v13 )
          v31 = 0;
        if ( j != v31 + 48 )
          break;
        v13 = *(_QWORD *)(v13 + 8);
        if ( v61 == v13 )
          goto LABEL_7;
        if ( !v13 )
          BUG();
      }
      v62 = j;
      if ( v61 == v13 )
        goto LABEL_7;
    }
    v60 = v13;
    v65 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
    while ( 1 )
    {
      v72[1] = 0;
      v73 = 0;
      v74 = 0;
      v72[0] = &unk_4A32910;
      v75 = 0;
      v76 = 0;
      sub_3103450((__int64)v72, (__int64)v22);
      if ( sub_3103560((__int64)v72, v15, v63, (__int64)v22, v23, v24) || (v66 = sub_98D040(v15, (__int64)v22)) != 0 )
      {
        v25 = v73;
        v72[0] = &unk_4A21008;
        if ( !v75 )
        {
          sub_C7D6A0((__int64)v73, 0, 8);
          goto LABEL_47;
        }
        v66 = 1;
        v64 = v22;
        v26 = &v73[2 * v75];
      }
      else
      {
        v25 = v73;
        v72[0] = &unk_4A21008;
        if ( !v75 )
        {
          sub_C7D6A0((__int64)v73, 0, 8);
          goto LABEL_37;
        }
        v64 = v22;
        v26 = &v73[2 * v75];
      }
      do
      {
        if ( *v25 != -4096 && *v25 != -8192 )
        {
          v27 = v25[1];
          if ( v27 )
          {
            if ( (v27 & 4) != 0 )
            {
              v28 = (unsigned __int64 *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
              v29 = (unsigned __int64)v28;
              if ( v28 )
              {
                if ( (unsigned __int64 *)*v28 != v28 + 2 )
                  _libc_free(*v28);
                j_j___libc_free_0(v29);
              }
            }
          }
        }
        v25 += 2;
      }
      while ( v25 != v26 );
      v22 = v64;
      sub_C7D6A0((__int64)v73, 16LL * v75, 8);
      if ( !v66 )
        goto LABEL_37;
LABEL_47:
      if ( !v71 )
      {
        ++v68;
        goto LABEL_72;
      }
      v32 = v71 - 1;
      v33 = 0;
      v34 = 1;
      v35 = (v71 - 1) & v65;
      v36 = &v69[7 * v35];
      v37 = *v36;
      if ( v15 != *v36 )
      {
        while ( v37 != -4096 )
        {
          if ( !v33 && v37 == -8192 )
            v33 = v36;
          v40 = v34 + 1;
          v34 = v35 + (unsigned int)v34;
          v35 = v32 & v34;
          v36 = &v69[7 * (v32 & (unsigned int)v34)];
          v37 = *v36;
          if ( v15 == *v36 )
            goto LABEL_49;
          LODWORD(v34) = v40;
        }
        if ( !v33 )
          v33 = v36;
        ++v68;
        v41 = v70 + 1;
        if ( 4 * ((int)v70 + 1) < 3 * v71 )
        {
          if ( v71 - HIDWORD(v70) - v41 <= v71 >> 3 )
          {
            sub_3104470((__int64)&v68, v71);
            if ( !v71 )
            {
LABEL_97:
              LODWORD(v70) = v70 + 1;
              BUG();
            }
            v47 = 1;
            v48 = (v71 - 1) & v65;
            v33 = &v69[7 * v48];
            v49 = *v33;
            v41 = v70 + 1;
            v46 = 0;
            if ( v15 != *v33 )
            {
              while ( v49 != -4096 )
              {
                if ( !v46 && v49 == -8192 )
                  v46 = v33;
                v50 = v47 + 1;
                v51 = (v71 - 1) & (v48 + v47);
                v48 = v51;
                v33 = &v69[7 * v51];
                v49 = *v33;
                if ( v15 == *v33 )
                  goto LABEL_68;
                v47 = v50;
              }
              goto LABEL_76;
            }
          }
          goto LABEL_68;
        }
LABEL_72:
        sub_3104470((__int64)&v68, 2 * v71);
        if ( !v71 )
          goto LABEL_97;
        v43 = (v71 - 1) & v65;
        v33 = &v69[7 * v43];
        v44 = *v33;
        v41 = v70 + 1;
        if ( v15 != *v33 )
        {
          v45 = 1;
          v46 = 0;
          while ( v44 != -4096 )
          {
            if ( !v46 && v44 == -8192 )
              v46 = v33;
            v54 = v45 + 1;
            v55 = (v71 - 1) & (v43 + v45);
            v43 = v55;
            v33 = &v69[7 * v55];
            v44 = *v33;
            if ( v15 == *v33 )
              goto LABEL_68;
            v45 = v54;
          }
LABEL_76:
          if ( v46 )
            v33 = v46;
        }
LABEL_68:
        LODWORD(v70) = v41;
        if ( *v33 != -4096 )
          --HIDWORD(v70);
        v42 = v33 + 3;
        *v33 = v15;
        v38 = 0;
        v39 = v33 + 1;
        *v39 = v42;
        v39[1] = 0x400000000LL;
        goto LABEL_51;
      }
LABEL_49:
      v38 = *((unsigned int *)v36 + 4);
      v39 = v36 + 1;
      if ( *((unsigned int *)v36 + 5) < (unsigned __int64)(v38 + 1) )
      {
        sub_C8D5F0((__int64)(v36 + 1), v36 + 3, v38 + 1, 8u, v38 + 1, v34);
        v38 = *((unsigned int *)v36 + 4);
      }
LABEL_51:
      *(_QWORD *)(*v39 + 8 * v38) = v22;
      ++*((_DWORD *)v39 + 2);
LABEL_37:
      v22 = (_QWORD *)*v22;
      if ( !v22 )
      {
        v13 = v60;
        goto LABEL_39;
      }
    }
  }
LABEL_7:
  sub_A68C30(a3, *a2, (__int64)&v67, 0, 0);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  v67 = off_4A32990;
  v8 = v71;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  if ( (_DWORD)v8 )
  {
    v9 = v69;
    v10 = &v69[7 * v8];
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[1];
        if ( (_QWORD *)v11 != v9 + 3 )
          _libc_free(v11);
      }
      v9 += 7;
    }
    while ( v10 != v9 );
    v8 = v71;
  }
  sub_C7D6A0((__int64)v69, 56 * v8, 8);
  nullsub_35();
  return a1;
}
