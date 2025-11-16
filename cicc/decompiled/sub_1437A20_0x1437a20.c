// Function: sub_1437A20
// Address: 0x1437a20
//
__int64 __fastcall sub_1437A20(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  void *v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // rsi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r8
  _QWORD *v20; // r12
  __int64 v21; // r14
  char v22; // al
  __int64 v23; // rbx
  _QWORD *v24; // r13
  _QWORD *v25; // rbx
  __int64 v26; // rax
  unsigned __int64 *v27; // rax
  __int64 j; // rcx
  __int64 v29; // rax
  unsigned int v30; // ecx
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // r9
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  unsigned __int64 v40; // rdi
  int v42; // r9d
  __int64 *v43; // rdx
  int v44; // edx
  unsigned int v45; // r10d
  int v46; // esi
  __int64 *v47; // rcx
  int v48; // r9d
  __int64 v49; // rsi
  int v50; // esi
  unsigned int v51; // r10d
  int i; // eax
  int v53; // r9d
  int v54; // r9d
  __int64 v55; // rsi
  __int64 v57; // [rsp+8h] [rbp-C8h]
  __int64 v58; // [rsp+10h] [rbp-C0h]
  __int64 v59; // [rsp+18h] [rbp-B8h]
  __int64 v60; // [rsp+20h] [rbp-B0h]
  __int64 v61; // [rsp+28h] [rbp-A8h]
  unsigned __int64 *v62; // [rsp+30h] [rbp-A0h]
  unsigned int v63; // [rsp+38h] [rbp-98h]
  char v64; // [rsp+3Fh] [rbp-91h]
  __int64 (__fastcall **v65)(); // [rsp+40h] [rbp-90h] BYREF
  __int64 v66; // [rsp+48h] [rbp-88h] BYREF
  _QWORD *v67; // [rsp+50h] [rbp-80h]
  __int64 v68; // [rsp+58h] [rbp-78h]
  unsigned int v69; // [rsp+60h] [rbp-70h]
  __int16 v70; // [rsp+70h] [rbp-60h] BYREF
  __int64 v71; // [rsp+78h] [rbp-58h]
  _QWORD *v72; // [rsp+80h] [rbp-50h]
  __int64 v73; // [rsp+88h] [rbp-48h]
  unsigned int v74; // [rsp+90h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_108:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9920C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_108;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9920C);
  v6 = *(__int64 **)(a1 + 8);
  v58 = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_109:
    BUG();
  v9 = &unk_4F9E06C;
  while ( *(_UNKNOWN **)v7 != &unk_4F9E06C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_109;
  }
  v10 = *(_QWORD **)(v7 + 8);
  v11 = (*(__int64 (__fastcall **)(_QWORD *, void *))(*v10 + 104LL))(v10, &unk_4F9E06C);
  v66 = 0;
  v67 = 0;
  v61 = v11 + 160;
  v13 = *(_QWORD *)(a2 + 80);
  v65 = off_49EB690;
  v68 = 0;
  v69 = 0;
  v59 = a2 + 72;
  if ( a2 + 72 == v13 )
  {
    v60 = 0;
    goto LABEL_16;
  }
  if ( !v13 )
    BUG();
  v12 = *(_QWORD *)(v13 + 24);
  if ( v12 != v13 + 16 )
  {
LABEL_15:
    v60 = v12;
LABEL_16:
    if ( v59 == v13 )
      goto LABEL_53;
    while ( 1 )
    {
      if ( !v60 )
        BUG();
      v14 = *(_DWORD *)(v58 + 184);
      if ( !v14 )
        goto LABEL_36;
      v15 = v14 - 1;
      v16 = *(_QWORD *)(v60 + 16);
      v10 = *(_QWORD **)(v58 + 168);
      v17 = v15 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v18 = &v10[2 * v17];
      v19 = *v18;
      if ( v16 != *v18 )
      {
        for ( i = 1; ; i = v53 )
        {
          if ( v19 == -8 )
            goto LABEL_36;
          v53 = i + 1;
          v17 = v15 & (i + v17);
          v18 = &v10[2 * v17];
          v19 = *v18;
          if ( v16 == *v18 )
            break;
        }
      }
      v20 = (_QWORD *)v18[1];
      if ( v20 )
        break;
LABEL_36:
      v9 = (void *)(a2 + 72);
      v12 = 0;
      for ( j = *(_QWORD *)(v60 + 8); ; j = *(_QWORD *)(v13 + 24) )
      {
        v29 = v13 - 24;
        if ( !v13 )
          v29 = 0;
        if ( j != v29 + 40 )
          break;
        v13 = *(_QWORD *)(v13 + 8);
        if ( v59 == v13 )
          goto LABEL_53;
        if ( !v13 )
          BUG();
      }
      v60 = j;
      if ( v59 == v13 )
        goto LABEL_53;
    }
    v57 = v13;
    v21 = v60 - 24;
    v63 = ((unsigned int)(v60 - 24) >> 9) ^ ((unsigned int)(v60 - 24) >> 4);
    while ( 1 )
    {
      v71 = 0;
      v70 = 0;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      sub_1436EA0((__int64)&v70, (__int64)v20);
      v22 = sub_1437020(v21, v61, (__int64)v20, (char *)&v70);
      v23 = v74;
      if ( v22 || (v35 = sub_14AE9E0(v21, v20), v23 = v74, (v64 = v35) != 0) )
      {
        if ( !(_DWORD)v23 )
        {
          j___libc_free_0(v72);
          goto LABEL_44;
        }
        v24 = v72;
        v64 = 1;
        v25 = &v72[2 * v23];
      }
      else
      {
        v24 = v72;
        v25 = &v72[2 * v74];
        if ( !v74 )
        {
          v10 = v72;
          j___libc_free_0(v72);
          goto LABEL_34;
        }
      }
      do
      {
        if ( *v24 != -8 && *v24 != -16 )
        {
          v26 = v24[1];
          if ( (v26 & 4) != 0 )
          {
            v27 = (unsigned __int64 *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v27 )
            {
              if ( (unsigned __int64 *)*v27 != v27 + 2 )
              {
                v62 = v27;
                _libc_free(*v27);
                v27 = v62;
              }
              j_j___libc_free_0(v27, 48);
            }
          }
        }
        v24 += 2;
      }
      while ( v25 != v24 );
      v10 = v72;
      j___libc_free_0(v72);
      if ( !v64 )
        goto LABEL_34;
LABEL_44:
      if ( !v69 )
      {
        ++v66;
        goto LABEL_72;
      }
      v10 = v67;
      v30 = (v69 - 1) & v63;
      v31 = &v67[7 * v30];
      v32 = *v31;
      if ( v21 != *v31 )
      {
        v42 = 1;
        v43 = 0;
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v43 )
            v43 = v31;
          v30 = (v69 - 1) & (v30 + v42);
          v31 = &v67[7 * v30];
          v32 = *v31;
          if ( v21 == *v31 )
            goto LABEL_46;
          ++v42;
        }
        if ( v43 )
          v31 = v43;
        ++v66;
        v44 = v68 + 1;
        if ( 4 * ((int)v68 + 1) < 3 * v69 )
        {
          if ( v69 - HIDWORD(v68) - v44 <= v69 >> 3 )
          {
            sub_1437700((__int64)&v66, v69);
            if ( !v69 )
            {
LABEL_105:
              LODWORD(v68) = v68 + 1;
              BUG();
            }
            v50 = 1;
            v51 = (v69 - 1) & v63;
            v31 = &v67[7 * v51];
            v10 = (_QWORD *)*v31;
            v44 = v68 + 1;
            v47 = 0;
            if ( v21 != *v31 )
            {
              while ( v10 != (_QWORD *)-8LL )
              {
                if ( !v47 && v10 == (_QWORD *)-16LL )
                  v47 = v31;
                v54 = v50 + 1;
                v55 = (v69 - 1) & (v51 + v50);
                v51 = v55;
                v31 = &v67[7 * v55];
                v10 = (_QWORD *)*v31;
                if ( v21 == *v31 )
                  goto LABEL_68;
                v50 = v54;
              }
              goto LABEL_84;
            }
          }
          goto LABEL_68;
        }
LABEL_72:
        sub_1437700((__int64)&v66, 2 * v69);
        if ( !v69 )
          goto LABEL_105;
        v45 = (v69 - 1) & v63;
        v31 = &v67[7 * v45];
        v10 = (_QWORD *)*v31;
        v44 = v68 + 1;
        if ( v21 != *v31 )
        {
          v46 = 1;
          v47 = 0;
          while ( v10 != (_QWORD *)-8LL )
          {
            if ( v10 == (_QWORD *)-16LL && !v47 )
              v47 = v31;
            v48 = v46 + 1;
            v49 = (v69 - 1) & (v45 + v46);
            v45 = v49;
            v31 = &v67[7 * v49];
            v10 = (_QWORD *)*v31;
            if ( v21 == *v31 )
              goto LABEL_68;
            v46 = v48;
          }
LABEL_84:
          if ( v47 )
            v31 = v47;
        }
LABEL_68:
        LODWORD(v68) = v44;
        if ( *v31 != -8 )
          --HIDWORD(v68);
        v34 = v31 + 3;
        *v31 = v21;
        v31[1] = (__int64)(v31 + 3);
        v31[2] = 0x400000000LL;
        goto LABEL_48;
      }
LABEL_46:
      v33 = *((unsigned int *)v31 + 4);
      if ( (unsigned int)v33 >= *((_DWORD *)v31 + 5) )
      {
        v10 = v31 + 1;
        sub_16CD150(v31 + 1, v31 + 3, 0, 8);
        v34 = (_QWORD *)(v31[1] + 8LL * *((unsigned int *)v31 + 4));
      }
      else
      {
        v34 = (_QWORD *)(v31[1] + 8 * v33);
      }
LABEL_48:
      *v34 = v20;
      ++*((_DWORD *)v31 + 4);
LABEL_34:
      v20 = (_QWORD *)*v20;
      if ( !v20 )
      {
        v13 = v57;
        goto LABEL_36;
      }
    }
  }
  while ( 1 )
  {
    v13 = *(_QWORD *)(v13 + 8);
    if ( v59 == v13 )
      break;
    if ( !v13 )
      BUG();
    v12 = *(_QWORD *)(v13 + 24);
    if ( v12 != v13 + 16 )
      goto LABEL_15;
  }
LABEL_53:
  v36 = sub_16BA580(v10, v9, v12);
  sub_1559E80(a2, v36, &v65, 0, 0, v37, a2, v57, v58);
  v65 = off_49EB690;
  if ( v69 )
  {
    v38 = v67;
    v39 = &v67[7 * v69];
    do
    {
      if ( *v38 != -16 && *v38 != -8 )
      {
        v40 = v38[1];
        if ( (_QWORD *)v40 != v38 + 3 )
          _libc_free(v40);
      }
      v38 += 7;
    }
    while ( v39 != v38 );
  }
  j___libc_free_0(v67);
  nullsub_544(&v65);
  return 0;
}
