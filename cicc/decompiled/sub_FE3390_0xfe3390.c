// Function: sub_FE3390
// Address: 0xfe3390
//
_QWORD *__fastcall sub_FE3390(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 *v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  int v8; // r12d
  unsigned int v9; // r15d
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rax
  char v18; // dl
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rcx
  __int64 *v31; // rax
  __int64 *v32; // rdx
  __int64 *v33; // rax
  __int64 *i; // rax
  __int64 v35; // r12
  __int64 v36; // r15
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 *v43; // rax
  __int64 v44; // rax
  _BYTE *v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rax
  _BYTE *v48; // rsi
  _QWORD *result; // rax
  __int64 v50; // r13
  __int64 v51; // r12
  __int64 j; // r13
  __int64 v53; // rbx
  __int64 *v54; // rdx
  __int64 *v55; // rdx
  __int64 v56; // rdi
  __int64 *v57; // rbx
  __int64 *v58; // r12
  __int64 v59; // rdi
  char v60; // dl
  __int64 *v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rdx
  __int64 v65; // [rsp+28h] [rbp-168h] BYREF
  __int64 v66; // [rsp+30h] [rbp-160h] BYREF
  int v67; // [rsp+38h] [rbp-158h]
  int v68; // [rsp+48h] [rbp-148h]
  __int64 v69; // [rsp+50h] [rbp-140h] BYREF
  __int64 v70; // [rsp+58h] [rbp-138h]
  __int64 *v71; // [rsp+60h] [rbp-130h]
  __int64 *v72; // [rsp+68h] [rbp-128h]
  _QWORD *v73; // [rsp+70h] [rbp-120h]
  __int64 **v74; // [rsp+78h] [rbp-118h]
  __int64 *v75; // [rsp+80h] [rbp-110h]
  __int64 *v76; // [rsp+88h] [rbp-108h]
  __int64 *v77; // [rsp+90h] [rbp-100h]
  _QWORD *v78; // [rsp+98h] [rbp-F8h]
  __int64 v79; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 *v80; // [rsp+A8h] [rbp-E8h]
  unsigned int v81; // [rsp+B0h] [rbp-E0h]
  unsigned int v82; // [rsp+B4h] [rbp-DCh]
  int v83; // [rsp+B8h] [rbp-D8h]
  char v84; // [rsp+BCh] [rbp-D4h]
  __int64 v85; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+100h] [rbp-90h] BYREF
  __int64 *v87; // [rsp+108h] [rbp-88h]
  __int64 v88; // [rsp+110h] [rbp-80h]
  int v89; // [rsp+118h] [rbp-78h]
  char v90; // [rsp+11Ch] [rbp-74h]
  char v91; // [rsp+120h] [rbp-70h] BYREF

  v71 = 0;
  v72 = 0;
  v73 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v70 = 8;
  v69 = sub_22077B0(64);
  v3 = (_QWORD *)(v69 + 24);
  v4 = (__int64 *)sub_22077B0(512);
  v74 = (__int64 **)(v69 + 24);
  v80 = &v85;
  v5 = *(_QWORD *)(a1 + 128);
  v6 = v4 + 64;
  *(_QWORD *)(v69 + 24) = v4;
  v7 = *(_QWORD *)(v5 + 80);
  v72 = v4;
  v73 = v4 + 64;
  v78 = v3;
  v76 = v4;
  if ( v7 )
    v7 -= 24;
  v77 = v4 + 64;
  v71 = v4;
  v81 = 8;
  v83 = 0;
  v84 = 1;
  if ( v4 )
    *v4 = v7;
  v85 = v7;
  v75 = v4 + 1;
  v82 = 1;
  v79 = 1;
  while ( 1 )
  {
    v65 = *v4;
    if ( v4 == v6 - 1 )
    {
      j_j___libc_free_0(v72, 512);
      v20 = (__int64)(*++v74 + 64);
      v72 = *v74;
      v73 = (_QWORD *)v20;
      v71 = v72;
    }
    else
    {
      v71 = v4 + 1;
    }
    sub_FDC9F0((__int64)&v86, &v65);
    v8 = v89;
    v9 = (unsigned int)v87;
    v10 = v86;
    if ( v89 != (_DWORD)v87 )
    {
      while ( 1 )
      {
        v11 = sub_B46EC0(v10, v9);
        v12 = *(_QWORD *)(a1 + 112);
        v66 = v11;
        if ( !(unsigned int)sub_FF0430(v12, v65, v11) )
          goto LABEL_15;
        if ( v84 )
        {
          v17 = v80;
          v13 = &v80[v82];
          if ( v80 != v13 )
          {
            while ( v66 != *v17 )
            {
              if ( v13 == ++v17 )
                goto LABEL_24;
            }
            goto LABEL_15;
          }
LABEL_24:
          if ( v82 < v81 )
          {
            ++v82;
            *v13 = v66;
            ++v79;
            goto LABEL_19;
          }
        }
        sub_C8CC70((__int64)&v79, v66, (__int64)v13, v14, v15, v16);
        if ( v18 )
        {
LABEL_19:
          v19 = v75;
          if ( v75 == v77 - 1 )
          {
            ++v9;
            sub_F9EB60(&v69, &v66);
            if ( v8 == v9 )
              break;
          }
          else
          {
            if ( v75 )
            {
              *v75 = v66;
              v19 = v75;
            }
            ++v9;
            v75 = v19 + 1;
            if ( v8 == v9 )
              break;
          }
        }
        else
        {
LABEL_15:
          if ( v8 == ++v9 )
            break;
        }
      }
    }
    v4 = v71;
    if ( v71 == v75 )
      break;
    v6 = v73;
  }
  v88 = 8;
  v87 = (__int64 *)&v91;
  v21 = *(_QWORD *)(a1 + 128);
  v86 = 0;
  v89 = 0;
  v22 = v21 + 72;
  v90 = 1;
  v23 = *(_QWORD *)(v21 + 80);
  if ( v21 + 72 == v23 )
    goto LABEL_115;
  do
  {
    v24 = 0;
    if ( v23 )
      v24 = v23 - 24;
    v65 = v24;
    v25 = v24;
    sub_FDC9F0((__int64)&v66, &v65);
    if ( v67 == v68 )
    {
      if ( v84 )
      {
        v28 = v80;
        v29 = &v80[v82];
        if ( v80 == v29 )
          goto LABEL_31;
        while ( v25 != *v28 )
        {
          if ( v29 == ++v28 )
            goto LABEL_31;
        }
      }
      else if ( !sub_C8CA60((__int64)&v79, v25) )
      {
        goto LABEL_31;
      }
      v30 = (__int64)v77;
      v31 = v75;
      v32 = v77 - 1;
      if ( v75 == v77 - 1 )
      {
        v63 = v78;
        if ( v73 - v71 + (((__int64 **)v78 - v74 - 1) << 6) + v75 - v76 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
        if ( (unsigned __int64)(v70 - (((__int64)v78 - v69) >> 3)) <= 1 )
        {
          sub_FE3210(&v69, 1u, 0);
          v63 = v78;
        }
        v63[1] = sub_22077B0(512);
        if ( v75 )
          *v75 = v25;
        v32 = (__int64 *)(*++v78 + 512LL);
        v76 = (__int64 *)*v78;
        v77 = v32;
        v75 = v76;
      }
      else
      {
        if ( v75 )
        {
          *v75 = v25;
          v31 = v75;
        }
        v75 = v31 + 1;
      }
      if ( !v90 )
        goto LABEL_114;
      v33 = v87;
      v32 = &v87[HIDWORD(v88)];
      if ( v87 == v32 )
      {
LABEL_48:
        if ( HIDWORD(v88) < (unsigned int)v88 )
        {
          ++HIDWORD(v88);
          *v32 = v25;
          ++v86;
          goto LABEL_31;
        }
LABEL_114:
        sub_C8CC70((__int64)&v86, v25, (__int64)v32, v30, v26, v27);
        goto LABEL_31;
      }
      while ( v25 != *v33 )
      {
        if ( v32 == ++v33 )
          goto LABEL_48;
      }
    }
LABEL_31:
    v23 = *(_QWORD *)(v23 + 8);
  }
  while ( v22 != v23 );
  for ( i = v71; v75 != v71; i = v71 )
  {
    while ( 1 )
    {
      v35 = *i;
      if ( i == v73 - 1 )
      {
        j_j___libc_free_0(v72, 512);
        v62 = (__int64)(*++v74 + 64);
        v72 = *v74;
        v73 = (_QWORD *)v62;
        v71 = v72;
      }
      else
      {
        v71 = i + 1;
      }
      v36 = *(_QWORD *)(v35 + 16);
      if ( v36 )
        break;
LABEL_64:
      i = v71;
      if ( v75 == v71 )
        goto LABEL_65;
    }
    do
    {
      v37 = *(_QWORD *)(v36 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v37 - 30) <= 0xAu )
      {
        while ( 1 )
        {
          v38 = *(_QWORD *)(a1 + 112);
          v66 = *(_QWORD *)(v37 + 40);
          if ( !(unsigned int)sub_FF0430(v38, v66, v35) )
            goto LABEL_61;
          if ( !v90 )
            goto LABEL_99;
          v43 = v87;
          v39 = &v87[HIDWORD(v88)];
          if ( v87 == v39 )
            break;
          while ( v66 != *v43 )
          {
            if ( v39 == ++v43 )
              goto LABEL_104;
          }
LABEL_61:
          v36 = *(_QWORD *)(v36 + 8);
          if ( !v36 )
            goto LABEL_64;
          while ( 1 )
          {
            v37 = *(_QWORD *)(v36 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v37 - 30) <= 0xAu )
              break;
            v36 = *(_QWORD *)(v36 + 8);
            if ( !v36 )
              goto LABEL_64;
          }
        }
LABEL_104:
        if ( HIDWORD(v88) >= (unsigned int)v88 )
        {
LABEL_99:
          sub_C8CC70((__int64)&v86, v66, (__int64)v39, v40, v41, v42);
          if ( !v60 )
            goto LABEL_61;
        }
        else
        {
          ++HIDWORD(v88);
          *v39 = v66;
          ++v86;
        }
        v61 = v75;
        if ( v75 == v77 - 1 )
        {
          sub_F9EB60(&v69, &v66);
        }
        else
        {
          if ( v75 )
          {
            *v75 = v66;
            v61 = v75;
          }
          v75 = v61 + 1;
        }
        goto LABEL_61;
      }
      v36 = *(_QWORD *)(v36 + 8);
    }
    while ( v36 );
  }
LABEL_65:
  v44 = *(_QWORD *)(a1 + 128);
  v45 = 0;
  v46 = v44 + 72;
  v47 = *(_QWORD *)(v44 + 80);
  if ( v46 == v47 )
  {
LABEL_115:
    v48 = 0;
    goto LABEL_68;
  }
  do
  {
    v47 = *(_QWORD *)(v47 + 8);
    ++v45;
  }
  while ( v46 != v47 );
  v48 = v45;
LABEL_68:
  result = (_QWORD *)sub_FDDB70(a2, (unsigned __int64)v48);
  v50 = *(_QWORD *)(a1 + 128);
  v51 = *(_QWORD *)(v50 + 80);
  for ( j = v50 + 72; j != v51; v51 = *(_QWORD *)(v51 + 8) )
  {
    v53 = v51 - 24;
    if ( !v51 )
      v53 = 0;
    if ( v84 )
    {
      result = v80;
      v54 = &v80[v82];
      if ( v80 == v54 )
        continue;
      while ( v53 != *result )
      {
        if ( v54 == ++result )
          goto LABEL_85;
      }
    }
    else
    {
      v48 = (_BYTE *)v53;
      result = sub_C8CA60((__int64)&v79, v53);
      if ( !result )
        continue;
    }
    if ( v90 )
    {
      result = v87;
      v55 = &v87[HIDWORD(v88)];
      if ( v87 != v55 )
      {
        while ( v53 != *result )
        {
          if ( v55 == ++result )
            goto LABEL_85;
        }
LABEL_81:
        v66 = v53;
        v48 = *(_BYTE **)(a2 + 8);
        if ( v48 == *(_BYTE **)(a2 + 16) )
        {
          result = sub_A413F0(a2, v48, &v66);
        }
        else
        {
          if ( v48 )
          {
            *(_QWORD *)v48 = v53;
            v48 = *(_BYTE **)(a2 + 8);
          }
          result = (_QWORD *)a2;
          v48 += 8;
          *(_QWORD *)(a2 + 8) = v48;
        }
      }
    }
    else
    {
      v48 = (_BYTE *)v53;
      result = sub_C8CA60((__int64)&v86, v53);
      if ( result )
        goto LABEL_81;
    }
LABEL_85:
    ;
  }
  if ( !v90 )
    result = (_QWORD *)_libc_free(v87, v48);
  if ( !v84 )
    result = (_QWORD *)_libc_free(v80, v48);
  v56 = v69;
  if ( v69 )
  {
    v57 = (__int64 *)v74;
    v58 = v78 + 1;
    if ( v78 + 1 > v74 )
    {
      do
      {
        v59 = *v57++;
        j_j___libc_free_0(v59, 512);
      }
      while ( v58 > v57 );
      v56 = v69;
    }
    return (_QWORD *)j_j___libc_free_0(v56, 8 * v70);
  }
  return result;
}
