// Function: sub_1165540
// Address: 0x1165540
//
__int64 __fastcall sub_1165540(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // r13d
  int v6; // ebx
  __int64 v7; // r15
  bool v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r14
  _BYTE *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // rcx
  __int64 v32; // rsi
  __int64 v33; // rsi
  int v34; // ebx
  unsigned int v35; // ebx
  unsigned __int8 *v36; // r15
  __int64 v37; // rax
  unsigned int v38; // r13d
  unsigned int v39; // eax
  unsigned int v40; // ebx
  __int64 v41; // rax
  unsigned int v42; // ebx
  bool v43; // al
  __int64 v44; // rbx
  _BYTE *v45; // rax
  unsigned __int8 *v46; // rdx
  unsigned int v47; // ebx
  __int64 v48; // rbx
  _BYTE *v49; // rax
  unsigned __int8 *v50; // rdx
  unsigned int v51; // ebx
  bool v52; // al
  bool v53; // bl
  unsigned int v54; // r12d
  unsigned __int8 *v55; // r15
  __int64 v56; // rax
  unsigned int v57; // ebx
  _QWORD *v58; // [rsp+8h] [rbp-88h]
  int v59; // [rsp+14h] [rbp-7Ch]
  __int64 v60; // [rsp+18h] [rbp-78h]
  _QWORD *v61; // [rsp+20h] [rbp-70h]
  _BYTE *v62; // [rsp+38h] [rbp-58h]
  _BYTE *v63; // [rsp+38h] [rbp-58h]
  _BYTE *v64; // [rsp+38h] [rbp-58h]
  __int64 v65; // [rsp+38h] [rbp-58h]
  __int64 v67; // [rsp+48h] [rbp-48h]
  __int64 v68; // [rsp+48h] [rbp-48h]
  int v69; // [rsp+48h] [rbp-48h]
  __int64 v70; // [rsp+48h] [rbp-48h]
  int v71; // [rsp+48h] [rbp-48h]
  __int64 v72[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 != 86 )
    return 0;
  if ( **(_BYTE **)(v2 - 64) > 0x15u )
    goto LABEL_76;
  v67 = *(_QWORD *)(v2 - 64);
  v6 = 2;
  v7 = -32;
  v8 = sub_AC30F0(v67);
  if ( v8 )
    goto LABEL_6;
  if ( *(_BYTE *)v67 == 17 )
  {
    v42 = *(_DWORD *)(v67 + 32);
    if ( v42 <= 0x40 )
      v43 = *(_QWORD *)(v67 + 24) == 0;
    else
      v43 = v42 == (unsigned int)sub_C444A0(v67 + 24);
  }
  else
  {
    v44 = *(_QWORD *)(v67 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 > 1 )
      goto LABEL_76;
    v45 = sub_AD7630(v67, 0, v67);
    v46 = (unsigned __int8 *)v67;
    if ( !v45 || *v45 != 17 )
    {
      if ( *(_BYTE *)(v44 + 8) == 17 )
      {
        v34 = *(_DWORD *)(v44 + 32);
        if ( v34 )
        {
          v69 = v34;
          v35 = 0;
          v36 = v46;
          do
          {
            v37 = sub_AD69F0(v36, v35);
            if ( !v37 )
              goto LABEL_76;
            if ( *(_BYTE *)v37 != 13 )
            {
              if ( *(_BYTE *)v37 != 17 )
                goto LABEL_76;
              v38 = *(_DWORD *)(v37 + 32);
              v8 = v38 <= 0x40 ? *(_QWORD *)(v37 + 24) == 0 : v38 == (unsigned int)sub_C444A0(v37 + 24);
              if ( !v8 )
                goto LABEL_76;
            }
            ++v35;
          }
          while ( v69 != v35 );
          if ( v8 )
            goto LABEL_89;
        }
      }
      goto LABEL_76;
    }
    v47 = *((_DWORD *)v45 + 8);
    if ( v47 <= 0x40 )
      v43 = *((_QWORD *)v45 + 3) == 0;
    else
      v43 = v47 == (unsigned int)sub_C444A0((__int64)(v45 + 24));
  }
  if ( v43 )
  {
LABEL_89:
    v7 = -32;
    v6 = 2;
    goto LABEL_6;
  }
LABEL_76:
  if ( **(_BYTE **)(v2 - 32) > 0x15u )
    return 0;
  v70 = *(_QWORD *)(v2 - 32);
  v6 = 1;
  v7 = -64;
  LOBYTE(v39) = sub_AC30F0(v70);
  v3 = v39;
  if ( (_BYTE)v39 )
  {
LABEL_6:
    v9 = *(_QWORD *)(v2 + v7);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v10 = *(_QWORD *)(a2 - 8);
    else
      v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v11 = *(_BYTE **)(v10 + 32);
    if ( v11 )
    {
      v12 = *(_QWORD *)(v10 + 40);
      **(_QWORD **)(v10 + 48) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v10 + 48);
    }
    *(_QWORD *)(v10 + 32) = v9;
    if ( v9 )
    {
      v13 = *(_QWORD *)(v9 + 16);
      *(_QWORD *)(v10 + 40) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = v10 + 40;
      *(_QWORD *)(v10 + 48) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v10 + 32;
    }
    if ( *v11 > 0x1Cu )
    {
      v72[0] = (__int64)v11;
      v62 = v11;
      v14 = *(_QWORD *)(a1 + 40) + 2096LL;
      sub_11604F0(v14, v72);
      v15 = *((_QWORD *)v62 + 2);
      if ( v15 )
      {
        if ( !*(_QWORD *)(v15 + 8) )
        {
          v72[0] = *(_QWORD *)(v15 + 24);
          sub_11604F0(v14, v72);
        }
      }
    }
    v68 = *(_QWORD *)(v2 - 96);
    if ( !*(_QWORD *)(v2 + 16) )
    {
      v41 = *(_QWORD *)(*(_QWORD *)(v2 - 96) + 16LL);
      if ( v41 )
      {
        if ( !*(_QWORD *)(v41 + 8) )
          return 1;
      }
    }
    v59 = v6;
    v61 = (_QWORD *)(a2 + 24);
    v58 = *(_QWORD **)(*(_QWORD *)(a2 + 40) + 56LL);
    v60 = *(_QWORD *)(v68 + 8);
    while ( 1 )
    {
      if ( v61 == v58 )
        return 1;
      v16 = (*v61 & 0xFFFFFFFFFFFFFFF8LL) == 0;
      v17 = (*v61 & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v61 = (_QWORD *)(*v61 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v16 )
        v17 = 0;
      if ( !(unsigned __int8)sub_98CD80((char *)v17) )
        return 1;
      v18 = 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
      {
        v19 = *(_QWORD *)(v17 - 8);
        v20 = v19 + v18;
      }
      else
      {
        v20 = v17;
        v19 = v17 - v18;
      }
      v21 = v19;
      if ( v20 != v19 )
        break;
LABEL_41:
      if ( v2 == v17 )
      {
        v29 = v68;
        if ( v68 == v2 )
          return 1;
        v2 = 0;
      }
      else if ( v68 == v17 )
      {
        v68 = 0;
        v29 = v2;
      }
      else
      {
        v29 = v2 | v68;
      }
      if ( !v29 )
        return 1;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = *(_BYTE **)v21;
        if ( v2 != *(_QWORD *)v21 )
          break;
        v23 = *(_QWORD *)(v2 + v7);
        if ( v2 )
        {
          v24 = *(_QWORD *)(v21 + 8);
          **(_QWORD **)(v21 + 16) = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = *(_QWORD *)(v21 + 16);
        }
        *(_QWORD *)v21 = v23;
        if ( v23 )
        {
          v25 = *(_QWORD *)(v23 + 16);
          *(_QWORD *)(v21 + 8) = v25;
          if ( v25 )
            *(_QWORD *)(v25 + 16) = v21 + 8;
          *(_QWORD *)(v21 + 16) = v23 + 16;
          *(_QWORD *)(v23 + 16) = v21;
        }
        v26 = *(_QWORD *)(a1 + 40);
        if ( *v22 <= 0x1Cu )
          goto LABEL_40;
        v72[0] = (__int64)v22;
        v63 = v22;
        sub_11604F0(v26 + 2096, v72);
        v27 = v26 + 2096;
        v28 = *((_QWORD *)v63 + 2);
        if ( v28 )
          goto LABEL_57;
LABEL_39:
        v26 = *(_QWORD *)(a1 + 40);
LABEL_40:
        v21 += 32;
        sub_F15FC0(v26, v17);
        if ( v20 == v21 )
          goto LABEL_41;
      }
      if ( (_BYTE *)v68 == v22 )
      {
        if ( v59 == 1 )
          v30 = sub_AD6400(v60);
        else
          v30 = sub_AD6450(v60);
        v31 = *(_BYTE **)v21;
        if ( *(_QWORD *)v21 )
        {
          v32 = *(_QWORD *)(v21 + 8);
          **(_QWORD **)(v21 + 16) = v32;
          if ( v32 )
            *(_QWORD *)(v32 + 16) = *(_QWORD *)(v21 + 16);
        }
        *(_QWORD *)v21 = v30;
        if ( v30 )
        {
          v33 = *(_QWORD *)(v30 + 16);
          *(_QWORD *)(v21 + 8) = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = v21 + 8;
          *(_QWORD *)(v21 + 16) = v30 + 16;
          *(_QWORD *)(v30 + 16) = v21;
        }
        v26 = *(_QWORD *)(a1 + 40);
        if ( *v31 <= 0x1Cu )
          goto LABEL_40;
        v72[0] = (__int64)v31;
        v64 = v31;
        sub_11604F0(v26 + 2096, v72);
        v27 = v26 + 2096;
        v28 = *((_QWORD *)v64 + 2);
        if ( !v28 )
          goto LABEL_39;
LABEL_57:
        if ( !*(_QWORD *)(v28 + 8) )
        {
          v72[0] = *(_QWORD *)(v28 + 24);
          sub_11604F0(v27, v72);
        }
        goto LABEL_39;
      }
      v21 += 32;
      if ( v20 == v21 )
        goto LABEL_41;
    }
  }
  if ( *(_BYTE *)v70 == 17 )
  {
    v40 = *(_DWORD *)(v70 + 32);
    if ( v40 > 0x40 )
    {
      if ( v40 != (unsigned int)sub_C444A0(v70 + 24) )
        return v3;
      goto LABEL_81;
    }
    v52 = *(_QWORD *)(v70 + 24) == 0;
    goto LABEL_103;
  }
  v48 = *(_QWORD *)(v70 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v48 + 8) - 17 > 1 )
    return v3;
  v49 = sub_AD7630(v70, 0, v70);
  v50 = (unsigned __int8 *)v70;
  if ( v49 && *v49 == 17 )
  {
    v51 = *((_DWORD *)v49 + 8);
    if ( v51 > 0x40 )
    {
      v52 = v51 == (unsigned int)sub_C444A0((__int64)(v49 + 24));
LABEL_103:
      if ( !v52 )
        return v3;
      goto LABEL_81;
    }
    if ( *((_QWORD *)v49 + 3) )
      return v3;
LABEL_81:
    v7 = -64;
    v6 = 1;
    goto LABEL_6;
  }
  if ( *(_BYTE *)(v48 + 8) == 17 )
  {
    v71 = *(_DWORD *)(v48 + 32);
    if ( v71 )
    {
      v65 = v2;
      v53 = 0;
      v54 = 0;
      v55 = v50;
      do
      {
        v56 = sub_AD69F0(v55, v54);
        if ( !v56 )
          return v3;
        if ( *(_BYTE *)v56 != 13 )
        {
          if ( *(_BYTE *)v56 != 17 )
            return v3;
          v57 = *(_DWORD *)(v56 + 32);
          v53 = v57 <= 0x40 ? *(_QWORD *)(v56 + 24) == 0 : v57 == (unsigned int)sub_C444A0(v56 + 24);
          if ( !v53 )
            return v3;
        }
        ++v54;
      }
      while ( v71 != v54 );
      v2 = v65;
      if ( v53 )
        goto LABEL_81;
    }
  }
  return v3;
}
