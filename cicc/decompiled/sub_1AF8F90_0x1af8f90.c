// Function: sub_1AF8F90
// Address: 0x1af8f90
//
_QWORD *__fastcall sub_1AF8F90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // rax
  __int64 v14; // r14
  _QWORD *v15; // rax
  _QWORD *v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r13
  _QWORD *v20; // rdx
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // rax
  __int64 *v24; // rsi
  __int64 v25; // rdx
  _QWORD *v26; // r14
  _BYTE *v27; // rdi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // r12
  __int64 v35; // r15
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // rcx
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rsi
  _QWORD *v43; // rdx
  __int64 v44; // rax
  _QWORD *v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+28h] [rbp-88h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  _BYTE *v51; // [rsp+30h] [rbp-80h] BYREF
  __int64 v52; // [rsp+38h] [rbp-78h]
  _BYTE v53[112]; // [rsp+40h] [rbp-70h] BYREF

  v13 = **(_QWORD **)(a1 + 32);
  v51 = v53;
  v52 = 0x800000000LL;
  v14 = *(_QWORD *)(v13 + 8);
  v49 = v13;
  if ( !v14 )
  {
LABEL_39:
    v24 = (__int64 *)v53;
    v25 = 0;
    goto LABEL_25;
  }
  while ( 1 )
  {
    v15 = sub_1648700(v14);
    if ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) <= 9u )
      break;
    v14 = *(_QWORD *)(v14 + 8);
    if ( !v14 )
      goto LABEL_39;
  }
LABEL_10:
  v19 = v15[5];
  v20 = *(_QWORD **)(a1 + 72);
  v17 = *(_QWORD **)(a1 + 64);
  if ( v20 != v17 )
  {
    v16 = &v20[*(unsigned int *)(a1 + 80)];
    v17 = sub_16CC9F0(a1 + 56, v19);
    if ( v19 == *v17 )
    {
      v32 = *(_QWORD *)(a1 + 72);
      if ( v32 == *(_QWORD *)(a1 + 64) )
        v33 = *(unsigned int *)(a1 + 84);
      else
        v33 = *(unsigned int *)(a1 + 80);
      v43 = (_QWORD *)(v32 + 8 * v33);
      goto LABEL_18;
    }
    v18 = *(_QWORD *)(a1 + 72);
    if ( v18 == *(_QWORD *)(a1 + 64) )
    {
      v17 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 84));
      v43 = v17;
      goto LABEL_18;
    }
    v17 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_7:
    if ( v17 != v16 )
      goto LABEL_8;
    goto LABEL_20;
  }
  v16 = &v17[*(unsigned int *)(a1 + 84)];
  if ( v17 == v16 )
  {
    v43 = *(_QWORD **)(a1 + 64);
  }
  else
  {
    do
    {
      if ( v19 == *v17 )
        break;
      ++v17;
    }
    while ( v16 != v17 );
    v43 = v16;
  }
LABEL_18:
  while ( v43 != v17 )
  {
    if ( *v17 < 0xFFFFFFFFFFFFFFFELL )
      goto LABEL_7;
    ++v17;
  }
  if ( v17 == v16 )
  {
LABEL_20:
    if ( *(_BYTE *)(sub_157EBA0(v19) + 16) == 28 )
    {
      v27 = v51;
      v26 = 0;
      goto LABEL_32;
    }
    v23 = (unsigned int)v52;
    if ( (unsigned int)v52 >= HIDWORD(v52) )
    {
      sub_16CD150((__int64)&v51, v53, 0, 8, v21, v22);
      v23 = (unsigned int)v52;
    }
    *(_QWORD *)&v51[8 * v23] = v19;
    LODWORD(v52) = v52 + 1;
    v14 = *(_QWORD *)(v14 + 8);
    if ( v14 )
      goto LABEL_9;
    goto LABEL_24;
  }
LABEL_8:
  while ( 1 )
  {
    v14 = *(_QWORD *)(v14 + 8);
    if ( !v14 )
      break;
LABEL_9:
    v15 = sub_1648700(v14);
    if ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) <= 9u )
      goto LABEL_10;
  }
LABEL_24:
  v24 = (__int64 *)v51;
  v25 = (unsigned int)v52;
LABEL_25:
  v26 = (_QWORD *)sub_1AAB350(v49, v24, v25, ".preheader", a2, a3, a5, a6, a7, a8, a9, a10, a11, a12, a4);
  if ( v26 )
  {
    v27 = v51;
    v28 = v26[3] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_DWORD)v52 )
    {
      v29 = v28 - 24;
      if ( !v28 )
        v29 = 0;
      v30 = v51;
      do
      {
        if ( *v30 == v29 )
          goto LABEL_32;
        ++v30;
      }
      while ( &v51[8 * (unsigned int)(v52 - 1) + 8] != (_BYTE *)v30 );
      v34 = 0;
      v50 = 8LL * (unsigned int)v52;
      do
      {
        v35 = *(_QWORD *)(*(_QWORD *)&v27[v34] + 32LL);
        if ( v35 == v26[7] + 72LL )
          goto LABEL_58;
        v36 = *(_QWORD **)(a1 + 64);
        if ( v35 )
          v35 -= 24;
        v37 = *(_QWORD **)(a1 + 72);
        if ( v37 == v36 )
        {
          v39 = &v36[*(unsigned int *)(a1 + 84)];
          if ( v36 == v39 )
          {
            v41 = *(_QWORD **)(a1 + 64);
          }
          else
          {
            do
            {
              if ( v35 == *v36 )
                break;
              ++v36;
            }
            while ( v39 != v36 );
            v41 = v39;
          }
        }
        else
        {
          v48 = &v37[*(unsigned int *)(a1 + 80)];
          v38 = sub_16CC9F0(a1 + 56, v35);
          v39 = v48;
          v36 = v38;
          if ( v35 == *v38 )
          {
            v44 = *(_QWORD *)(a1 + 72);
            if ( v44 == *(_QWORD *)(a1 + 64) )
            {
              v27 = v51;
              v41 = (_QWORD *)(v44 + 8LL * *(unsigned int *)(a1 + 84));
              goto LABEL_53;
            }
            v41 = (_QWORD *)(v44 + 8LL * *(unsigned int *)(a1 + 80));
          }
          else
          {
            v40 = *(_QWORD *)(a1 + 72);
            if ( v40 == *(_QWORD *)(a1 + 64) )
            {
              v27 = v51;
              v36 = (_QWORD *)(v40 + 8LL * *(unsigned int *)(a1 + 84));
              v41 = v36;
              goto LABEL_53;
            }
            v36 = (_QWORD *)(v40 + 8LL * *(unsigned int *)(a1 + 80));
            v41 = v36;
          }
          v27 = v51;
        }
LABEL_53:
        while ( v41 != v36 && *v36 >= 0xFFFFFFFFFFFFFFFELL )
          ++v36;
        if ( v36 != v39 )
        {
          v42 = *(_QWORD *)&v27[v34];
          if ( v42 )
            goto LABEL_56;
          break;
        }
LABEL_58:
        v34 += 8;
      }
      while ( v50 != v34 );
    }
    v42 = *(_QWORD *)v27;
LABEL_56:
    sub_1580AC0(v26, v42);
  }
  v27 = v51;
LABEL_32:
  if ( v27 != v53 )
    _libc_free((unsigned __int64)v27);
  return v26;
}
