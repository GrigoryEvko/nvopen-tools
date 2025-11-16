// Function: sub_3856BA0
// Address: 0x3856ba0
//
bool __fastcall sub_3856BA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // r12
  int v12; // eax
  int v13; // edx
  __int64 v14; // rcx
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 *v18; // r12
  unsigned int v19; // ebx
  int v20; // r8d
  int v21; // edx
  _QWORD *v22; // rax
  __int64 **v23; // r13
  __int64 v24; // rax
  unsigned __int64 *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // r12
  unsigned __int64 *v30; // r9
  int v31; // ecx
  unsigned __int64 *v32; // rdx
  unsigned __int64 *v33; // rax
  unsigned __int64 v34; // rcx
  int v35; // ebx
  bool result; // al
  int v37; // eax
  int v38; // esi
  __int64 v39; // rcx
  unsigned int v40; // edx
  __int64 v41; // rax
  __int64 *v42; // rdi
  __int64 v43; // rcx
  __int64 *v44; // rax
  bool v45; // cc
  __int64 v46; // rbx
  int v47; // eax
  int v48; // r9d
  __int64 v49; // rax
  unsigned __int64 *v50; // rax
  unsigned __int64 v51; // rdi
  unsigned int v52; // edx
  unsigned __int64 v53; // rsi
  int v54; // eax
  int v55; // r9d
  __int64 v56; // rax
  __int64 v57; // [rsp+0h] [rbp-A0h]
  __int64 v58; // [rsp+18h] [rbp-88h] BYREF
  __int64 v59[4]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 *v60; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v61; // [rsp+48h] [rbp-58h] BYREF
  _DWORD v62[20]; // [rsp+50h] [rbp-50h] BYREF

  v60 = (unsigned __int64 *)v62;
  v61 = 0x200000000LL;
  v7 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v8 = *(_QWORD *)(a2 - 8);
    v9 = v8 + v7;
  }
  else
  {
    v8 = a2 - v7;
    v9 = a2;
  }
  if ( v8 != v9 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)v8;
      if ( *(_BYTE *)(*(_QWORD *)v8 + 16LL) > 0x10u )
      {
        v12 = *(_DWORD *)(a1 + 160);
        if ( !v12 )
          goto LABEL_12;
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 144);
        v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v11 != *v16 )
        {
          v47 = 1;
          while ( v17 != -8 )
          {
            v48 = v47 + 1;
            v49 = v13 & (v15 + v47);
            v15 = v49;
            v16 = (__int64 *)(v14 + 16 * v49);
            v17 = *v16;
            if ( v11 == *v16 )
              goto LABEL_11;
            v47 = v48;
          }
          goto LABEL_12;
        }
LABEL_11:
        v11 = v16[1];
        if ( !v11 )
          goto LABEL_12;
      }
      v10 = (unsigned int)v61;
      if ( (unsigned int)v61 >= HIDWORD(v61) )
      {
        sub_16CD150((__int64)&v60, v62, 0, 8, a5, (int)&v60);
        v10 = (unsigned int)v61;
      }
      v8 += 24;
      v60[v10] = v11;
      LODWORD(v61) = v61 + 1;
      if ( v9 == v8 )
      {
        v50 = v60;
        goto LABEL_44;
      }
    }
  }
  v50 = (unsigned __int64 *)v62;
LABEL_44:
  v46 = sub_15A3BA0(*v50, *(__int64 ***)a2, 0);
  if ( v46 )
  {
    v59[0] = a2;
    sub_38526A0(a1 + 136, v59)[1] = v46;
    result = 1;
    if ( v60 != (unsigned __int64 *)v62 )
    {
      _libc_free((unsigned __int64)v60);
      return 1;
    }
    return result;
  }
LABEL_12:
  if ( v60 != (unsigned __int64 *)v62 )
    _libc_free((unsigned __int64)v60);
  v18 = *(__int64 **)(a2 - 24);
  v19 = sub_16431D0(*v18);
  if ( (unsigned int)sub_15A9570(*(_QWORD *)(a1 + 40), *(_QWORD *)a2) >= v19 )
  {
    v37 = *(_DWORD *)(a1 + 256);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 240);
      v40 = (v37 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v41 = v39 + 32LL * v40;
      v42 = *(__int64 **)v41;
      if ( v18 != *(__int64 **)v41 )
      {
        v54 = 1;
        while ( v42 != (__int64 *)-8LL )
        {
          v55 = v54 + 1;
          v40 = v38 & (v54 + v40);
          v41 = v39 + 32LL * v40;
          v42 = *(__int64 **)v41;
          if ( v18 == *(__int64 **)v41 )
            goto LABEL_35;
          v54 = v55;
        }
        goto LABEL_15;
      }
LABEL_35:
      v43 = *(_QWORD *)(v41 + 8);
      v60 = (unsigned __int64 *)v43;
      v62[0] = *(_DWORD *)(v41 + 24);
      if ( v62[0] > 0x40u )
      {
        sub_16A4FD0((__int64)&v61, (const void **)(v41 + 16));
        if ( !v60 )
          goto LABEL_40;
        goto LABEL_37;
      }
      v61 = *(_QWORD *)(v41 + 16);
      if ( v43 )
      {
LABEL_37:
        v59[0] = a2;
        v44 = sub_3854530(a1 + 232, v59);
        v45 = *((_DWORD *)v44 + 6) <= 0x40u;
        v44[1] = (__int64)v60;
        if ( v45 && v62[0] <= 0x40u )
        {
          v51 = v61;
          v44[2] = v61;
          v52 = v62[0];
          *((_DWORD *)v44 + 6) = v62[0];
          v53 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v52;
          if ( v52 > 0x40 )
          {
            v56 = (unsigned int)(((unsigned __int64)v52 + 63) >> 6) - 1;
            *(_QWORD *)(v51 + 8 * v56) &= v53;
          }
          else
          {
            v44[2] = v51 & v53;
          }
        }
        else
        {
          sub_16A51C0((__int64)(v44 + 2), (__int64)&v61);
        }
LABEL_40:
        if ( v62[0] > 0x40u && v61 )
          j_j___libc_free_0_0(v61);
      }
    }
  }
LABEL_15:
  v21 = *(_DWORD *)(a1 + 184);
  v59[0] = 0;
  v59[1] = -1;
  v59[2] = 0;
  v59[3] = 0;
  if ( v21 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, (__int64)v18, &v58, v59) )
  {
    v60 = (unsigned __int64 *)a2;
    v22 = sub_176FB00(a1 + 168, (__int64 *)&v60);
    v22[1] = v58;
  }
  v23 = *(__int64 ***)a1;
  v24 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v25 = *(unsigned __int64 **)(a2 - 8);
    v26 = (__int64)&v25[v24];
  }
  else
  {
    v25 = (unsigned __int64 *)(a2 - v24 * 8);
    v26 = a2;
  }
  v27 = v26 - (_QWORD)v25;
  v60 = (unsigned __int64 *)v62;
  v28 = 0xAAAAAAAAAAAAAAABLL * (v27 >> 3);
  v61 = 0x400000000LL;
  v29 = v28;
  if ( (unsigned __int64)v27 > 0x60 )
  {
    v57 = v27;
    sub_16CD150((__int64)&v60, v62, 0xAAAAAAAAAAAAAAABLL * (v27 >> 3), 8, v20, (int)&v60);
    v30 = v60;
    v31 = v61;
    v27 = v57;
    v32 = &v60[(unsigned int)v61];
  }
  else
  {
    v30 = (unsigned __int64 *)v62;
    v31 = 0;
    v32 = (unsigned __int64 *)v62;
  }
  if ( v27 > 0 )
  {
    v33 = v25;
    do
    {
      v34 = *v33;
      ++v32;
      v33 += 3;
      *(v32 - 1) = v34;
      --v29;
    }
    while ( v29 );
    v30 = v60;
    v31 = v61;
  }
  LODWORD(v61) = v31 + v28;
  v35 = sub_14A5330(v23, a2, (__int64)v30, (unsigned int)(v31 + v28));
  if ( v60 != (unsigned __int64 *)v62 )
    _libc_free((unsigned __int64)v60);
  return v35 == 0;
}
