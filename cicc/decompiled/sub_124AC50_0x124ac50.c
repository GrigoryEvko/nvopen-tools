// Function: sub_124AC50
// Address: 0x124ac50
//
__int64 __fastcall sub_124AC50(
        __int64 a1,
        unsigned __int8 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int8 v8; // al
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // r12
  __int64 v12; // rdx
  _QWORD *v13; // rcx
  char v14; // al
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r13
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *i; // rdx
  unsigned __int64 v23; // rax
  int v24; // r9d
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *j; // rdx
  unsigned int v28; // r14d
  __int64 v29; // r12
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned int v37; // edx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned int v40; // eax
  unsigned __int16 v41; // ax
  unsigned __int8 v42; // al
  unsigned __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // r8
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rcx
  char v49; // al
  unsigned int v50; // eax
  int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r8
  char v60; // [rsp+0h] [rbp-B0h]
  char v61; // [rsp+0h] [rbp-B0h]
  char v62; // [rsp+0h] [rbp-B0h]
  __int64 v63; // [rsp+0h] [rbp-B0h]
  __int64 v64; // [rsp+0h] [rbp-B0h]
  __int64 v65; // [rsp+0h] [rbp-B0h]
  __int64 v66; // [rsp+0h] [rbp-B0h]
  __int64 v67; // [rsp+8h] [rbp-A8h]
  __int16 v69; // [rsp+16h] [rbp-9Ah]
  unsigned __int64 v70; // [rsp+28h] [rbp-88h] BYREF
  char v71; // [rsp+30h] [rbp-80h]
  _BYTE *v72; // [rsp+38h] [rbp-78h] BYREF
  __int64 v73; // [rsp+40h] [rbp-70h]
  __int64 v74; // [rsp+48h] [rbp-68h]
  _BYTE v75[96]; // [rsp+50h] [rbp-60h] BYREF

  *(_QWORD *)(a1 + 88) = a3;
  v70 = a3;
  v69 = a4;
  v8 = sub_C5F410(&a7, &v70, 0, a4, a5);
  v11 = v8;
  *(_BYTE *)a1 = v8;
  if ( !byte_4F92488 && (unsigned int)sub_2207590(&byte_4F92488) )
  {
    sub_12499A0(&qword_4F92490, (__int64)&v70, v32, v33, v9);
    __cxa_atexit((void (*)(void *))sub_12496B0, &qword_4F92490, &qword_4A427C0);
    sub_2207640(&byte_4F92488);
  }
  v12 = qword_4F92490;
  v13 = (_QWORD *)0x8E38E38E38E38E39LL;
  if ( 0x8E38E38E38E38E39LL * ((qword_4F92498 - qword_4F92490) >> 3) > v11 )
  {
    v31 = qword_4F92490 + 72 * v11;
    v13 = &v72;
    v14 = *(_BYTE *)v31;
    v72 = v75;
    v73 = 0;
    v71 = v14;
    v74 = 40;
    if ( *(_QWORD *)(v31 + 16) )
    {
      sub_12495E0((__int64)&v72, v31 + 8, qword_4F92490, (__int64)&v72, v9, v10);
      v14 = v71;
    }
  }
  else
  {
    v71 = 0;
    v73 = 0;
    v74 = 40;
    v72 = v75;
    v14 = 0;
  }
  *(_BYTE *)(a1 + 8) = v14;
  v67 = a1 + 16;
  sub_12495E0(a1 + 16, (__int64)&v72, v12, (__int64)v13, v9, v10);
  if ( v72 != v75 )
    _libc_free(v72, &v72);
  if ( !*(_BYTE *)(a1 + 8) )
    return 0;
  v18 = *(_QWORD *)(a1 + 24);
  v19 = *(unsigned int *)(a1 + 104);
  if ( v18 != v19 )
  {
    if ( v18 < v19 )
    {
      *(_DWORD *)(a1 + 104) = v18;
      v19 = v18;
    }
    else
    {
      if ( v18 > *(unsigned int *)(a1 + 108) )
      {
        sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), *(_QWORD *)(a1 + 24), 8u, v15, v16);
        v19 = *(unsigned int *)(a1 + 104);
      }
      v20 = *(_QWORD *)(a1 + 96);
      v21 = (_QWORD *)(v20 + 8 * v19);
      for ( i = (_QWORD *)(v20 + 8 * v18); i != v21; ++v21 )
      {
        if ( v21 )
          *v21 = 0;
      }
      *(_DWORD *)(a1 + 104) = v18;
      v19 = *(_QWORD *)(a1 + 24);
    }
  }
  v23 = *(unsigned int *)(a1 + 168);
  if ( v19 != v23 )
  {
    v24 = v19;
    if ( v19 < v23 )
    {
      *(_DWORD *)(a1 + 168) = v19;
      v23 = v19;
    }
    else
    {
      if ( v19 > *(unsigned int *)(a1 + 172) )
      {
        sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v19, 8u, v15, (unsigned int)v19);
        v23 = *(unsigned int *)(a1 + 168);
        v24 = v19;
      }
      v25 = *(_QWORD *)(a1 + 160);
      v26 = (_QWORD *)(v25 + 8 * v23);
      for ( j = (_QWORD *)(v25 + 8 * v19); j != v26; ++v26 )
      {
        if ( v26 )
          *v26 = 0;
      }
      *(_DWORD *)(a1 + 168) = v24;
      v23 = *(_QWORD *)(a1 + 24);
    }
  }
  v28 = 0;
  v29 = 0;
  if ( v23 )
  {
    while ( 2 )
    {
      v30 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + v29);
      switch ( *(_BYTE *)(*(_QWORD *)(a1 + 16) + v29) & 0x7F )
      {
        case 0:
          v62 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + v29);
          v42 = sub_C5F410(&a7, &v70, 0, v30, v15);
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v42;
          if ( v62 < 0 )
            *(_QWORD *)(v15 + *(_QWORD *)(a1 + 96)) = *(char *)(v15 + *(_QWORD *)(a1 + 96));
          goto LABEL_35;
        case 1:
          v61 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + v29);
          v41 = sub_C5F510((__int64)&a7, &v70, 0, v30, v15);
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v41;
          if ( v61 < 0 )
            *(_QWORD *)(v15 + *(_QWORD *)(a1 + 96)) = *(__int16 *)(v15 + *(_QWORD *)(a1 + 96));
          goto LABEL_35;
        case 2:
          v60 = *(_BYTE *)(*(_QWORD *)(a1 + 16) + v29);
          v40 = sub_C5F610((__int64)&a7, &v70, 0, v30, v15);
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v40;
          if ( v60 < 0 )
            *(_QWORD *)(v15 + *(_QWORD *)(a1 + 96)) = *(int *)(v15 + *(_QWORD *)(a1 + 96));
          goto LABEL_35;
        case 3:
          v39 = sub_C5F710((__int64)&a7, &v70, 0, v30, v15);
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v39;
          goto LABEL_35;
        case 4:
          if ( (v30 & 0x80u) == 0LL )
            *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = sub_C5EE20(&a7, (unsigned __int64)&v70, 0);
          else
            *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = sub_C5F090(&a7, (unsigned __int64)&v70, 0);
          v15 = 8 * v29;
          goto LABEL_35;
        case 5:
          v37 = a2;
          goto LABEL_43;
        case 6:
          if ( !HIBYTE(v69) )
            return 0;
          if ( (_BYTE)v69 )
          {
            if ( (_BYTE)v69 != 1 )
LABEL_73:
              BUG();
            v37 = 8;
          }
          else
          {
            v37 = 4;
          }
LABEL_43:
          v38 = sub_C5F720(&a7, &v70, v37, 0, v15);
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v38;
          goto LABEL_35;
        case 7:
          if ( !v28 )
            return 0;
          v36 = v70;
          v15 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v70;
          v70 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * (v28 - 1)) + v36;
          goto LABEL_35;
        case 8:
          goto LABEL_34;
        case 9:
          v43 = sub_C5EE20(&a7, (unsigned __int64)&v70, 0);
          v45 = 8 * v29;
          *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v43;
          v46 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29);
          if ( !byte_4F92468 )
          {
            v65 = v45;
            v51 = sub_2207590(&byte_4F92468);
            v45 = v65;
            if ( v51 )
            {
              qword_4F92470 = 0;
              qword_4F92478 = 0;
              qword_4F92480 = 0;
              sub_1249720(&qword_4F92470, 2u, v52, v53, v65);
              v71 = 5;
              v74 = 40;
              v72 = v75;
              v75[0] = 9;
              v73 = 1;
              v54 = qword_4F92470 + 80;
              *(_BYTE *)(qword_4F92470 + 72) = 5;
              sub_12495E0(v54, (__int64)&v72, v55, v56, v57, v58);
              v59 = v65;
              if ( v72 != v75 )
              {
                _libc_free(v72, &v72);
                v59 = v65;
              }
              v66 = v59;
              __cxa_atexit((void (*)(void *))sub_12496B0, &qword_4F92470, &qword_4A427C0);
              sub_2207640(&byte_4F92468);
              v45 = v66;
            }
          }
          v47 = (unsigned int)v46;
          v48 = qword_4F92470;
          if ( (unsigned int)v46 < 0x8E38E38E38E38E39LL * ((qword_4F92478 - qword_4F92470) >> 3) )
          {
            v47 = qword_4F92470 + 72LL * (unsigned int)v46;
            v49 = *(_BYTE *)v47;
            v72 = v75;
            v73 = 0;
            v71 = v49;
            v74 = 40;
            if ( *(_QWORD *)(v47 + 16) )
            {
              v64 = v45;
              sub_12495E0((__int64)&v72, v47 + 8, v47, qword_4F92470, v45, v44);
              v49 = v71;
              v45 = v64;
            }
          }
          else
          {
            v71 = 0;
            v73 = 0;
            v74 = 40;
            v72 = v75;
            v49 = 0;
          }
          *(_BYTE *)(a1 + 8) = v49;
          v63 = v45;
          sub_12495E0(v67, (__int64)&v72, v47, v48, v45, v44);
          v15 = v63;
          if ( v72 != v75 )
          {
            _libc_free(v72, &v72);
            v15 = v63;
          }
          if ( !*(_BYTE *)(a1 + 8) )
            return 0;
          goto LABEL_35;
        case 0x1E:
          v34 = **(_QWORD **)(a1 + 96);
          if ( v34 == 3 )
          {
            v50 = sub_C5F610((__int64)&a7, &v70, 0, v30, v15);
            v15 = 8 * v29;
            *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v50;
          }
          else
          {
            if ( v34 > 4 )
              return 0;
LABEL_34:
            v35 = sub_C5EE20(&a7, (unsigned __int64)&v70, 0);
            v15 = 8 * v29;
            *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v29) = v35;
          }
LABEL_35:
          v29 = ++v28;
          *(_QWORD *)(*(_QWORD *)(a1 + 160) + v15) = v70;
          if ( (unsigned __int64)v28 >= *(_QWORD *)(a1 + 24) )
            break;
          continue;
        default:
          goto LABEL_73;
      }
      break;
    }
  }
  *(_QWORD *)(a1 + 88) = v70;
  return 1;
}
