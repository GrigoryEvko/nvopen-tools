// Function: sub_117BAB0
// Address: 0x117bab0
//
unsigned __int8 *__fastcall sub_117BAB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  char v4; // al
  char v5; // r13
  bool v6; // zf
  char v8; // si
  __int64 v9; // rcx
  _BYTE *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rcx
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  _BYTE *v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  void **v22; // rax
  __int64 v23; // rdx
  char v24; // cl
  void **v25; // rsi
  __int64 v26; // r14
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rax
  bool v32; // al
  __int64 v33; // r12
  void **v34; // rax
  char v35; // dl
  void **v36; // rsi
  unsigned int v37; // r12d
  void **v38; // rax
  void **v39; // rsi
  char v40; // al
  _BYTE *v41; // rsi
  unsigned int v42; // edx
  void **v43; // rax
  void **v44; // rsi
  char v45; // al
  unsigned int v46; // edx
  void *v47; // rax
  _BYTE *v48; // rsi
  int v49; // [rsp+Ch] [rbp-74h]
  int v50; // [rsp+Ch] [rbp-74h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  _BYTE *v52; // [rsp+18h] [rbp-68h]
  char v53; // [rsp+18h] [rbp-68h]
  unsigned __int64 v54; // [rsp+20h] [rbp-60h]
  __int64 v55; // [rsp+20h] [rbp-60h]
  void **v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+20h] [rbp-60h]
  unsigned int v58; // [rsp+20h] [rbp-60h]
  __int64 v59; // [rsp+28h] [rbp-58h]
  char v60; // [rsp+28h] [rbp-58h]
  void **v61; // [rsp+28h] [rbp-58h]
  char v62; // [rsp+28h] [rbp-58h]
  unsigned __int64 v63; // [rsp+38h] [rbp-48h]
  _QWORD v64[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = 0;
  v4 = sub_920620(a1);
  if ( !v4 )
    return (unsigned __int8 *)v2;
  v5 = v4;
  if ( ((*(_BYTE *)(a1 + 1) >> 1) & 0xA) != 0xA )
    return (unsigned __int8 *)v2;
  v6 = *(_BYTE *)a1 == 86;
  v64[0] = a2;
  v64[1] = a1;
  if ( !v6 )
    return 0;
  v8 = *(_BYTE *)(a1 + 7);
  if ( (v8 & 0x40) != 0 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v10 = *(_BYTE **)v9;
  v11 = *(_QWORD *)(*(_QWORD *)v9 + 16LL);
  if ( !v11
    || *(_QWORD *)(v11 + 8)
    || *v10 != 83
    || (v57 = *((_QWORD *)v10 - 8)) == 0
    || (v26 = *((_QWORD *)v10 - 4)) == 0 )
  {
LABEL_9:
    if ( (v8 & 0x40) != 0 )
      v12 = *(_QWORD *)(a1 - 8);
    else
      v12 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v13 = *(_BYTE **)v12;
    v14 = *(_QWORD *)(*(_QWORD *)v12 + 16LL);
    if ( !v14 )
      return 0;
    if ( *(_QWORD *)(v14 + 8) )
      return 0;
    if ( *v13 != 83 )
      return 0;
    v59 = *((_QWORD *)v13 - 8);
    if ( !v59 )
      return 0;
    v15 = *((_QWORD *)v13 - 4);
    if ( !v15 )
      return 0;
    v54 = sub_B53900((__int64)v13) & 0xFFFFFFFFFFLL | v51 & 0xFFFFFF0000000000LL;
    v16 = sub_986520(a1);
    v17 = *(_BYTE **)(v16 + 32);
    if ( *v17 > 0x15u )
      return 0;
    v18 = *(_QWORD *)(v16 + 64);
    v19 = *(_QWORD *)(v18 + 16);
    if ( !v19 )
      return 0;
    v2 = *(_QWORD *)(v19 + 8);
    if ( v2 || *(_BYTE *)v18 <= 0x1Cu )
      return 0;
    if ( sub_B52830(v54) )
      return (unsigned __int8 *)v2;
    if ( *(_BYTE *)v15 == 18 )
    {
      if ( *(void **)(v15 + 24) == sub_C33340() )
        v20 = *(_QWORD *)(v15 + 32);
      else
        v20 = v15 + 24;
      if ( (*(_BYTE *)(v20 + 20) & 7) != 3 )
        return (unsigned __int8 *)v2;
    }
    else
    {
      v21 = *(_QWORD *)(v15 + 8);
      v55 = v21;
      if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 || *(_BYTE *)v15 > 0x15u )
        return (unsigned __int8 *)v2;
      v22 = (void **)sub_AD7630(v15, 0, v21);
      v23 = v55;
      v24 = 0;
      if ( !v22 || (v56 = v22, *(_BYTE *)v22 != 18) )
      {
        if ( *(_BYTE *)(v23 + 8) == 17 )
        {
          v50 = *(_DWORD *)(v23 + 32);
          if ( v50 )
          {
            v42 = 0;
            while ( 1 )
            {
              v53 = v24;
              v58 = v42;
              v43 = (void **)sub_AD69F0((unsigned __int8 *)v15, v42);
              v44 = v43;
              if ( !v43 )
                break;
              v45 = *(_BYTE *)v43;
              v46 = v58;
              v24 = v53;
              if ( v45 != 13 )
              {
                if ( v45 != 18 )
                  return (unsigned __int8 *)v2;
                v47 = sub_C33340();
                v46 = v58;
                v48 = v44[3] == v47 ? v44[4] : v44 + 3;
                if ( (v48[20] & 7) != 3 )
                  return (unsigned __int8 *)v2;
                v24 = v5;
              }
              v42 = v46 + 1;
              if ( v50 == v42 )
              {
                if ( v24 )
                  goto LABEL_25;
                return (unsigned __int8 *)v2;
              }
            }
          }
        }
        return (unsigned __int8 *)v2;
      }
      if ( v22[3] == sub_C33340() )
        v25 = (void **)v56[4];
      else
        v25 = v56 + 3;
      if ( (*((_BYTE *)v25 + 20) & 7) != 3 )
        return (unsigned __int8 *)v2;
    }
LABEL_25:
    if ( *(_BYTE *)v18 == 43 && v59 == *(_QWORD *)(v18 - 64) && v17 == *(_BYTE **)(v18 - 32) )
      return sub_1178690((__int64)v64, v59, v15, (unsigned __int8 *)v18, (__int64)v17, 1);
    return (unsigned __int8 *)v2;
  }
  v63 = sub_B53900((__int64)v10);
  v8 = *(_BYTE *)(a1 + 7);
  v51 = v63 & 0xFFFFFFFFFFLL;
  if ( (v8 & 0x40) != 0 )
    v27 = *(_QWORD *)(a1 - 8);
  else
    v27 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v28 = *(_QWORD *)(v27 + 32);
  v29 = *(_QWORD *)(v28 + 16);
  if ( !v29
    || (v2 = *(_QWORD *)(v29 + 8)) != 0
    || *(_BYTE *)v28 <= 0x1Cu
    || (v52 = *(_BYTE **)(sub_986520(a1) + 64), *v52 > 0x15u) )
  {
    if ( *(_BYTE *)a1 != 86 )
      return 0;
    goto LABEL_9;
  }
  LOBYTE(v30) = sub_B52830(v63);
  if ( (_BYTE)v30 )
    return (unsigned __int8 *)v2;
  if ( *(_BYTE *)v26 == 18 )
  {
    if ( *(void **)(v26 + 24) == sub_C33340() )
      v31 = *(_QWORD *)(v26 + 32);
    else
      v31 = v26 + 24;
    v32 = (*(_BYTE *)(v31 + 20) & 7) == 3;
    goto LABEL_58;
  }
  v33 = *(_QWORD *)(v26 + 8);
  v60 = v30;
  if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 > 1 || *(_BYTE *)v26 > 0x15u )
    return (unsigned __int8 *)v2;
  v34 = (void **)sub_AD7630(v26, 0, v30);
  v35 = v60;
  if ( v34 )
  {
    v61 = v34;
    if ( *(_BYTE *)v34 == 18 )
    {
      if ( v34[3] == sub_C33340() )
        v36 = (void **)v61[4];
      else
        v36 = v61 + 3;
      v32 = (*((_BYTE *)v36 + 20) & 7) == 3;
LABEL_58:
      if ( v32 )
      {
LABEL_59:
        if ( *(_BYTE *)v28 == 43 && v57 == *(_QWORD *)(v28 - 64) && v52 == *(_BYTE **)(v28 - 32) )
          return sub_1178690((__int64)v64, v57, v26, (unsigned __int8 *)v28, (__int64)v52, 0);
      }
      return (unsigned __int8 *)v2;
    }
  }
  if ( *(_BYTE *)(v33 + 8) == 17 )
  {
    v49 = *(_DWORD *)(v33 + 32);
    if ( v49 )
    {
      v37 = 0;
      while ( 1 )
      {
        v62 = v35;
        v38 = (void **)sub_AD69F0((unsigned __int8 *)v26, v37);
        v39 = v38;
        if ( !v38 )
          break;
        v40 = *(_BYTE *)v38;
        v35 = v62;
        if ( v40 != 13 )
        {
          if ( v40 != 18 )
            return (unsigned __int8 *)v2;
          v41 = v39[3] == sub_C33340() ? v39[4] : v39 + 3;
          if ( (v41[20] & 7) != 3 )
            return (unsigned __int8 *)v2;
          v35 = v5;
        }
        if ( v49 == ++v37 )
        {
          if ( v35 )
            goto LABEL_59;
          return (unsigned __int8 *)v2;
        }
      }
    }
  }
  return (unsigned __int8 *)v2;
}
