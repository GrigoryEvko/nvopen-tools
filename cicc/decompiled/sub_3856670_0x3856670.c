// Function: sub_3856670
// Address: 0x3856670
//
bool __fastcall sub_3856670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
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
  unsigned int v18; // ebx
  __int64 v19; // rax
  int v20; // r8d
  int v21; // eax
  __int64 v22; // rcx
  int v23; // edi
  __int64 v24; // rsi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 *v29; // rax
  bool v30; // cc
  int v31; // edx
  _QWORD *v32; // rax
  __int64 **v33; // r13
  __int64 v34; // r15
  unsigned __int64 *v35; // rbx
  __int64 v36; // r15
  __int64 v37; // r15
  unsigned __int64 v38; // r10
  unsigned __int64 v39; // r12
  unsigned __int64 *v40; // r9
  int v41; // ecx
  unsigned __int64 *v42; // rdx
  unsigned __int64 *v43; // rax
  unsigned __int64 v44; // rcx
  int v45; // ebx
  bool result; // al
  __int64 v47; // rbx
  int v48; // eax
  int v49; // r9d
  __int64 v50; // rax
  unsigned __int64 *v51; // rax
  int v52; // eax
  int v53; // r10d
  unsigned __int64 v54; // rsi
  unsigned int v55; // edx
  unsigned __int64 v56; // rdi
  __int64 v57; // rax
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
          v48 = 1;
          while ( v17 != -8 )
          {
            v49 = v48 + 1;
            v50 = v13 & (v15 + v48);
            v15 = v50;
            v16 = (__int64 *)(v14 + 16 * v50);
            v17 = *v16;
            if ( v11 == *v16 )
              goto LABEL_11;
            v48 = v49;
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
        v51 = v60;
        goto LABEL_46;
      }
    }
  }
  v51 = (unsigned __int64 *)v62;
LABEL_46:
  v47 = sub_15A4180(*v51, *(__int64 ***)a2, 0);
  if ( v47 )
  {
    v59[0] = a2;
    sub_38526A0(a1 + 136, v59)[1] = v47;
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
  v18 = sub_16431D0(*(_QWORD *)a2);
  v19 = **(_QWORD **)(a2 - 24);
  if ( *(_BYTE *)(v19 + 8) == 16 )
    v19 = **(_QWORD **)(v19 + 16);
  if ( v18 >= 8 * (unsigned int)sub_15A9520(*(_QWORD *)(a1 + 40), *(_DWORD *)(v19 + 8) >> 8) )
  {
    v21 = *(_DWORD *)(a1 + 256);
    if ( v21 )
    {
      v22 = *(_QWORD *)(a2 - 24);
      v23 = v21 - 1;
      v24 = *(_QWORD *)(a1 + 240);
      v25 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v26 = (__int64 *)(v24 + 32LL * v25);
      v27 = *v26;
      if ( v22 != *v26 )
      {
        v52 = 1;
        while ( v27 != -8 )
        {
          v53 = v52 + 1;
          v25 = v23 & (v52 + v25);
          v26 = (__int64 *)(v24 + 32LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
            goto LABEL_19;
          v52 = v53;
        }
        goto LABEL_27;
      }
LABEL_19:
      v28 = v26[1];
      v60 = (unsigned __int64 *)v28;
      v62[0] = *((_DWORD *)v26 + 6);
      if ( v62[0] > 0x40u )
      {
        sub_16A4FD0((__int64)&v61, (const void **)v26 + 2);
        if ( !v60 )
          goto LABEL_24;
        goto LABEL_21;
      }
      v61 = v26[2];
      if ( v28 )
      {
LABEL_21:
        v59[0] = a2;
        v29 = sub_3854530(a1 + 232, v59);
        v30 = *((_DWORD *)v29 + 6) <= 0x40u;
        v29[1] = (__int64)v60;
        if ( v30 && v62[0] <= 0x40u )
        {
          v54 = v61;
          v29[2] = v61;
          v55 = v62[0];
          *((_DWORD *)v29 + 6) = v62[0];
          v56 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v55;
          if ( v55 > 0x40 )
          {
            v57 = (unsigned int)(((unsigned __int64)v55 + 63) >> 6) - 1;
            *(_QWORD *)(v54 + 8 * v57) &= v56;
          }
          else
          {
            v29[2] = v56 & v54;
          }
        }
        else
        {
          sub_16A51C0((__int64)(v29 + 2), (__int64)&v61);
        }
LABEL_24:
        if ( v62[0] > 0x40u && v61 )
          j_j___libc_free_0_0(v61);
      }
    }
  }
LABEL_27:
  v31 = *(_DWORD *)(a1 + 184);
  v59[0] = 0;
  v59[1] = -1;
  v59[2] = 0;
  v59[3] = 0;
  if ( v31 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, *(_QWORD *)(a2 - 24), &v58, v59) )
  {
    v60 = (unsigned __int64 *)a2;
    v32 = sub_176FB00(a1 + 168, (__int64 *)&v60);
    v32[1] = v58;
  }
  v33 = *(__int64 ***)a1;
  v34 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v35 = *(unsigned __int64 **)(a2 - 8);
    v36 = (__int64)&v35[v34];
  }
  else
  {
    v35 = (unsigned __int64 *)(a2 - v34 * 8);
    v36 = a2;
  }
  v37 = v36 - (_QWORD)v35;
  v60 = (unsigned __int64 *)v62;
  v38 = 0xAAAAAAAAAAAAAAABLL * (v37 >> 3);
  v61 = 0x400000000LL;
  v39 = v38;
  if ( (unsigned __int64)v37 > 0x60 )
  {
    sub_16CD150((__int64)&v60, v62, 0xAAAAAAAAAAAAAAABLL * (v37 >> 3), 8, v20, (int)&v60);
    v40 = v60;
    v41 = v61;
    v38 = 0xAAAAAAAAAAAAAAABLL * (v37 >> 3);
    v42 = &v60[(unsigned int)v61];
  }
  else
  {
    v40 = (unsigned __int64 *)v62;
    v41 = 0;
    v42 = (unsigned __int64 *)v62;
  }
  if ( v37 > 0 )
  {
    v43 = v35;
    do
    {
      v44 = *v43;
      ++v42;
      v43 += 3;
      *(v42 - 1) = v44;
      --v39;
    }
    while ( v39 );
    v40 = v60;
    v41 = v61;
  }
  LODWORD(v61) = v38 + v41;
  v45 = sub_14A5330(v33, a2, (__int64)v40, (unsigned int)(v38 + v41));
  if ( v60 != (unsigned __int64 *)v62 )
    _libc_free((unsigned __int64)v60);
  return v45 == 0;
}
