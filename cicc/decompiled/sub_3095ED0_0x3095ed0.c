// Function: sub_3095ED0
// Address: 0x3095ed0
//
__int64 __fastcall sub_3095ED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 i; // rbx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // r15
  int v16; // ebx
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 j; // rbx
  char v21; // dl
  __int64 v22; // r8
  __int64 v23; // r14
  __int64 *v24; // r15
  unsigned __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rsi
  __int64 v31; // rax
  int v32; // r14d
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rax
  _QWORD **v39; // r14
  _QWORD **v40; // r15
  _QWORD *v41; // r12
  __int64 *v42; // rdx
  __int64 *v43; // rax
  unsigned __int64 *v44; // rdi
  unsigned __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 *v49; // rax
  _QWORD *v51; // [rsp+10h] [rbp-190h]
  _QWORD *v52; // [rsp+18h] [rbp-188h]
  __int64 v53; // [rsp+18h] [rbp-188h]
  _QWORD *v54; // [rsp+20h] [rbp-180h]
  unsigned __int64 v55; // [rsp+28h] [rbp-178h]
  unsigned __int8 v56; // [rsp+37h] [rbp-169h]
  _BYTE *v57; // [rsp+40h] [rbp-160h] BYREF
  unsigned int v58; // [rsp+48h] [rbp-158h]
  unsigned int v59; // [rsp+4Ch] [rbp-154h]
  _BYTE v60[128]; // [rsp+50h] [rbp-150h] BYREF
  __int64 v61; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 *v62; // [rsp+D8h] [rbp-C8h]
  __int64 v63; // [rsp+E0h] [rbp-C0h]
  int v64; // [rsp+E8h] [rbp-B8h]
  char v65; // [rsp+ECh] [rbp-B4h]
  _BYTE v66[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v6 = (__int64 *)v66;
  v7 = *(_QWORD *)(a2 + 320);
  v61 = 0;
  v8 = *(_QWORD *)(a2 + 32);
  v62 = (__int64 *)v66;
  v63 = 16;
  v64 = 0;
  v65 = 1;
  v51 = (_QWORD *)(a2 + 320);
  v52 = (_QWORD *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) != a2 + 320 )
  {
    v56 = 0;
LABEL_3:
    v9 = v52[6];
    v54 = v52 + 6;
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 44) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v11 )
      {
        v11 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
          break;
      }
    }
LABEL_9:
    if ( v54 == (_QWORD *)v11 )
      goto LABEL_24;
LABEL_10:
    v13 = *(_QWORD *)(v11 + 32);
    v14 = v13 + 40LL * (*(_DWORD *)(v11 + 40) & 0xFFFFFF);
    if ( v14 == v13 )
      goto LABEL_17;
    v55 = v11;
    v15 = *(_QWORD *)(v11 + 32);
LABEL_12:
    if ( *(_BYTE *)v15 )
      goto LABEL_15;
    if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
      goto LABEL_15;
    v16 = *(_DWORD *)(v15 + 8);
    if ( v16 >= 0 )
      goto LABEL_15;
    while ( 1 )
    {
      v26 = sub_2EBEE10(v8, v16);
      v30 = v26;
      if ( !v26
        || *(_WORD *)(v26 + 68) != 20 && (*(_BYTE *)(*(_QWORD *)(v26 + 16) + 32LL) & 0x10) == 0
        || (v31 = *(_QWORD *)(v26 + 32), *(_BYTE *)(v31 + 40))
        || (v32 = *(_DWORD *)(v31 + 48), v32 >= 0) )
      {
        if ( v16 != *(_DWORD *)(v15 + 8) )
        {
          sub_2EAB0C0(v15, v16);
          v56 = 1;
        }
LABEL_15:
        v15 += 40;
        if ( v14 == v15 )
        {
          v11 = v55;
LABEL_17:
          v17 = (_QWORD *)(*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
          v18 = v17;
          if ( !v17 )
            BUG();
          v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
          v19 = *v17;
          if ( (v19 & 4) != 0 || (*((_BYTE *)v18 + 44) & 4) == 0 )
            goto LABEL_9;
          for ( j = v19; ; j = *(_QWORD *)v11 )
          {
            v11 = j & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
              break;
          }
          if ( v54 == (_QWORD *)v11 )
          {
LABEL_24:
            v52 = (_QWORD *)(*v52 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v51 == v52 )
            {
              v6 = v62;
              v21 = v65;
              goto LABEL_26;
            }
            goto LABEL_3;
          }
          goto LABEL_10;
        }
        goto LABEL_12;
      }
      if ( !v65 )
        break;
      v33 = v62;
      v28 = HIDWORD(v63);
      v27 = &v62[HIDWORD(v63)];
      if ( v62 == v27 )
      {
LABEL_51:
        if ( HIDWORD(v63) >= (unsigned int)v63 )
          break;
        v16 = v32;
        ++HIDWORD(v63);
        *v27 = v30;
        ++v61;
      }
      else
      {
        while ( v30 != *v33 )
        {
          if ( v27 == ++v33 )
            goto LABEL_51;
        }
LABEL_37:
        v16 = v32;
      }
    }
    sub_C8CC70((__int64)&v61, v30, (__int64)v27, v28, v29, a6);
    goto LABEL_37;
  }
  v56 = 0;
  v21 = 1;
LABEL_26:
  v59 = 16;
  v57 = v60;
  v58 = 0;
  if ( !v21 )
    goto LABEL_73;
LABEL_27:
  v22 = (__int64)&v6[HIDWORD(v63)];
LABEL_28:
  if ( v6 != (__int64 *)v22 )
  {
    while ( 1 )
    {
      v23 = *v6;
      v24 = v6;
      if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( (__int64 *)v22 == ++v6 )
        goto LABEL_31;
    }
    if ( (__int64 *)v22 != v6 )
    {
      v34 = 0;
      v35 = *(unsigned int *)(*(_QWORD *)(v23 + 32) + 8LL);
      if ( (int)v35 >= 0 )
      {
LABEL_55:
        v36 = *(_QWORD *)(*(_QWORD *)(v8 + 304) + 8 * v35);
        goto LABEL_56;
      }
      while ( 1 )
      {
        v36 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16 * (v35 & 0x7FFFFFFF) + 8);
LABEL_56:
        if ( !v36 )
          goto LABEL_76;
        if ( (*(_BYTE *)(v36 + 3) & 0x10) != 0 )
          break;
LABEL_58:
        v37 = v24 + 1;
        if ( v24 + 1 == (__int64 *)v22 )
          goto LABEL_61;
        while ( 1 )
        {
          v23 = *v37;
          v24 = v37;
          if ( (unsigned __int64)*v37 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v22 == ++v37 )
            goto LABEL_61;
        }
        if ( (__int64 *)v22 == v37 )
        {
LABEL_61:
          v38 = 8LL * (unsigned int)v34;
          v39 = (_QWORD **)&v57[v38];
          if ( &v57[v38] != v57 )
          {
            v40 = (_QWORD **)v57;
            do
            {
              v41 = *v40;
              if ( v65 )
              {
                v42 = &v62[HIDWORD(v63)];
                v43 = v62;
                if ( v62 != v42 )
                {
                  while ( v41 != (_QWORD *)*v43 )
                  {
                    if ( v42 == ++v43 )
                      goto LABEL_69;
                  }
                  --HIDWORD(v63);
                  *v43 = v62[HIDWORD(v63)];
                  ++v61;
                }
              }
              else
              {
                v49 = sub_C8CA60((__int64)&v61, (__int64)v41);
                if ( v49 )
                {
                  *v49 = -2;
                  ++v64;
                  ++v61;
                }
              }
LABEL_69:
              ++v40;
              sub_2E31080(v41[3] + 40LL, (__int64)v41);
              v44 = (unsigned __int64 *)v41[1];
              v45 = *v41 & 0xFFFFFFFFFFFFFFF8LL;
              *v44 = v45 | *v44 & 7;
              *(_QWORD *)(v45 + 8) = v44;
              *v41 &= 7uLL;
              v41[1] = 0;
              sub_2E790D0(a2, (__int64)v41, v45, v46, v47, v48);
            }
            while ( v39 != v40 );
            v56 = 1;
            LODWORD(v34) = v58;
          }
          if ( (_DWORD)v34 )
          {
            v6 = v62;
            v58 = 0;
            if ( !v65 )
            {
LABEL_73:
              v22 = (__int64)&v6[(unsigned int)v63];
              goto LABEL_28;
            }
            goto LABEL_27;
          }
          goto LABEL_31;
        }
        v35 = *(unsigned int *)(*(_QWORD *)(v23 + 32) + 8LL);
        if ( (int)v35 >= 0 )
          goto LABEL_55;
      }
      while ( 1 )
      {
        v36 = *(_QWORD *)(v36 + 32);
        if ( !v36 )
          break;
        if ( (*(_BYTE *)(v36 + 3) & 0x10) == 0 )
          goto LABEL_58;
      }
LABEL_76:
      if ( v34 + 1 > (unsigned __int64)v59 )
      {
        v53 = v22;
        sub_C8D5F0((__int64)&v57, v60, v34 + 1, 8u, v22, a6);
        v34 = v58;
        v22 = v53;
      }
      *(_QWORD *)&v57[8 * v34] = v23;
      v34 = ++v58;
      goto LABEL_58;
    }
  }
LABEL_31:
  if ( v57 != v60 )
    _libc_free((unsigned __int64)v57);
  if ( !v65 )
    _libc_free((unsigned __int64)v62);
  return v56;
}
