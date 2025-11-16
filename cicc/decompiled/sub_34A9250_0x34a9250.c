// Function: sub_34A9250
// Address: 0x34a9250
//
void __fastcall sub_34A9250(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int *a7,
        __int64 a8,
        __int64 a9,
        char a10,
        __int64 a11)
{
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r12
  int v15; // ebx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r9
  unsigned int *v26; // rbx
  __int64 v27; // r14
  unsigned __int64 v28; // r8
  __int64 v29; // r12
  unsigned __int64 v30; // r13
  unsigned int v31; // eax
  unsigned int v32; // esi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r8
  _BYTE *v38; // rdi
  __int64 v39; // rax
  __int64 *v40; // rax
  unsigned __int64 v41; // r14
  __int64 v42; // r8
  bool v43; // r14
  unsigned __int64 v44; // [rsp+10h] [rbp-160h]
  __int64 v45; // [rsp+18h] [rbp-158h]
  __int64 v46; // [rsp+20h] [rbp-150h]
  unsigned int *v47; // [rsp+30h] [rbp-140h]
  int v49; // [rsp+40h] [rbp-130h]
  unsigned int v50; // [rsp+48h] [rbp-128h]
  int v51; // [rsp+5Ch] [rbp-114h] BYREF
  unsigned int *v52; // [rsp+60h] [rbp-110h] BYREF
  __int64 v53; // [rsp+68h] [rbp-108h]
  _BYTE v54[16]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v55; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE *v56; // [rsp+88h] [rbp-E8h]
  __int64 v57; // [rsp+90h] [rbp-E0h]
  _BYTE v58[72]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v59; // [rsp+E0h] [rbp-90h] BYREF
  char *v60; // [rsp+E8h] [rbp-88h]
  __int64 v61; // [rsp+F0h] [rbp-80h]
  char v62; // [rsp+F8h] [rbp-78h] BYREF
  __int64 v63; // [rsp+100h] [rbp-70h]

  v11 = a1[1];
  v12 = v11 + 816;
  v13 = v11 + 224;
  if ( (unsigned int)(*(_DWORD *)(*a1 + 56LL) - 2) > 1 )
    v12 = v13;
  v46 = v12;
  if ( (*(_BYTE *)(v12 + 8) & 1) != 0 )
  {
    v14 = v12 + 16;
    v15 = 7;
  }
  else
  {
    v14 = *(_QWORD *)(v12 + 16);
    v22 = *(unsigned int *)(v12 + 24);
    v18 = v14;
    if ( !(_DWORD)v22 )
      goto LABEL_60;
    v15 = v22 - 1;
  }
  v62 = 0;
  v59 = 0;
  v63 = 0;
  v51 = 0;
  if ( a10 )
    v51 = (unsigned __int16)a9 | ((_DWORD)a8 << 16);
  v55 = a11;
  v52 = a7;
  v49 = 1;
  v50 = v15 & sub_F11290((__int64 *)&v52, &v51, &v55);
  while ( 1 )
  {
    v16 = v14 + 72LL * v50;
    if ( a7 == *(unsigned int **)v16
      && a10 == *(_BYTE *)(v16 + 24)
      && (!a10 || a8 == *(_QWORD *)(v16 + 8) && a9 == *(_QWORD *)(v16 + 16))
      && a11 == *(_QWORD *)(v16 + 32) )
    {
      v45 = v14 + 72LL * v50;
      if ( (*(_BYTE *)(v46 + 8) & 1) != 0 )
        goto LABEL_12;
      v18 = *(_QWORD *)(v46 + 16);
      goto LABEL_58;
    }
    if ( sub_F34140(v16, (__int64)&v59) )
      break;
    v50 = v15 & (v49 + v50);
    ++v49;
  }
  if ( (*(_BYTE *)(v46 + 8) & 1) != 0 )
  {
    v45 = v46 + 592;
LABEL_12:
    v17 = 576;
    v18 = v46 + 16;
    goto LABEL_13;
  }
  v18 = *(_QWORD *)(v46 + 16);
  v22 = *(unsigned int *)(v46 + 24);
LABEL_60:
  v45 = v18 + 72 * v22;
LABEL_58:
  v17 = 72LL * *(unsigned int *)(v46 + 24);
LABEL_13:
  if ( v45 != v17 + v18 )
  {
    v52 = (unsigned int *)v54;
    v53 = 0x200000000LL;
    v19 = *(unsigned int *)(v45 + 48);
    if ( (_DWORD)v19 )
    {
      sub_349DD80((__int64)&v52, v45 + 40, v19, v45, a5, a6);
      v26 = v52;
      v47 = &v52[2 * (unsigned int)v53];
      if ( v47 != v52 )
      {
        do
        {
          v27 = a1[1];
          v28 = v26[1] | ((unsigned __int64)*v26 << 32);
          v56 = v58;
          v29 = v27 + 16;
          v30 = v28;
          v55 = v27 + 16;
          v57 = 0x400000000LL;
          v31 = *(_DWORD *)(v27 + 208);
          if ( v31 )
          {
            sub_34A3C90((__int64)&v55, v28, v23, v24, v28, v25);
          }
          else
          {
            v32 = *(_DWORD *)(v27 + 212);
            if ( v32 )
            {
              v23 = v27 + 24;
              while ( v28 > *(_QWORD *)v23 )
              {
                ++v31;
                v23 += 16;
                if ( v32 == v31 )
                  goto LABEL_29;
              }
              v32 = v31;
            }
LABEL_29:
            sub_34A26E0((__int64)&v55, v32, v23, v24, v28, v25);
          }
          v59 = v27 + 16;
          v60 = &v62;
          v61 = 0x400000000LL;
          sub_34A26E0((__int64)&v59, *(_DWORD *)(v27 + 212), v33, v34, v35, v36);
          v25 = (__int64)v60;
          if ( (_DWORD)v57 && (v38 = v56, v24 = *((unsigned int *)v56 + 2), *((_DWORD *)v56 + 3) < (unsigned int)v24) )
          {
            v39 = (__int64)&v56[16 * (unsigned int)v57 - 16];
            v23 = (__int64)&v60[16 * (unsigned int)v61 - 16];
            v24 = *(unsigned int *)(v23 + 12);
            if ( *(_DWORD *)(v39 + 12) != (_DWORD)v24 )
            {
              if ( v60 != &v62 )
              {
                _libc_free((unsigned __int64)v60);
                v38 = v56;
                v39 = (__int64)&v56[16 * (unsigned int)v57 - 16];
              }
              goto LABEL_35;
            }
            v24 = *(_QWORD *)v23;
            v43 = *(_QWORD *)v39 == *(_QWORD *)v23;
          }
          else
          {
            v43 = 1;
            if ( (_DWORD)v61 )
              v43 = *((_DWORD *)v60 + 3) >= *((_DWORD *)v60 + 2);
          }
          if ( v60 != &v62 )
            _libc_free((unsigned __int64)v60);
          v38 = v56;
          if ( v43 )
            goto LABEL_41;
          v39 = (__int64)&v56[16 * (unsigned int)v57 - 16];
LABEL_35:
          v23 = *(_QWORD *)v39;
          v40 = (__int64 *)(*(_QWORD *)v39 + 16LL * *(unsigned int *)(v39 + 12));
          if ( v30 >= *v40 )
          {
            v41 = v40[1];
            v44 = *v40;
            sub_34A3230((__int64)&v55, *v40, v23, v24, v37);
            if ( v30 > v44 )
              sub_34A8ED0(v29, v44, v30 - 1, 0, v42, v25);
            if ( v30 < v41 )
              sub_34A8ED0(v29, v30 + 1, v41, 0, v42, v25);
            v38 = v56;
          }
LABEL_41:
          if ( v38 != v58 )
            _libc_free((unsigned __int64)v38);
          v26 += 2;
        }
        while ( v47 != v26 );
      }
    }
    v20 = *(_QWORD *)(v45 + 40);
    if ( v20 != v45 + 56 )
      _libc_free(v20);
    *(_QWORD *)v45 = 0;
    *(_QWORD *)(v45 + 8) = 0;
    *(_QWORD *)(v45 + 16) = 0;
    *(_BYTE *)(v45 + 24) = 1;
    *(_QWORD *)(v45 + 32) = 0;
    v21 = *(_DWORD *)(v46 + 8);
    ++*(_DWORD *)(v46 + 12);
    *(_DWORD *)(v46 + 8) = (2 * (v21 >> 1) - 2) | v21 & 1;
    if ( v52 != (unsigned int *)v54 )
      _libc_free((unsigned __int64)v52);
  }
}
