// Function: sub_282FEA0
// Address: 0x282fea0
//
__int64 __fastcall sub_282FEA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r11d
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 v20; // dl
  __int64 v21; // rax
  __int64 v22; // r10
  char v23; // al
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r10
  __int64 v27; // rdx
  char *v28; // rax
  char *v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rdi
  char *v33; // rax
  unsigned int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // [rsp+18h] [rbp-158h]
  __int64 v43; // [rsp+18h] [rbp-158h]
  __int64 v44; // [rsp+20h] [rbp-150h]
  __int64 v45; // [rsp+20h] [rbp-150h]
  __int64 v46; // [rsp+28h] [rbp-148h]
  unsigned __int8 v48; // [rsp+38h] [rbp-138h]
  unsigned __int8 v49; // [rsp+38h] [rbp-138h]
  _QWORD v50[2]; // [rsp+40h] [rbp-130h] BYREF
  __int64 v51; // [rsp+50h] [rbp-120h]
  int v52; // [rsp+58h] [rbp-118h]
  __int64 v53; // [rsp+60h] [rbp-110h]
  __int64 v54; // [rsp+68h] [rbp-108h]
  _BYTE *v55; // [rsp+70h] [rbp-100h]
  __int64 v56; // [rsp+78h] [rbp-F8h]
  _BYTE v57[16]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+90h] [rbp-E0h] BYREF
  _QWORD v59[2]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-C8h]
  __int64 v61; // [rsp+B0h] [rbp-C0h]
  __int64 v62; // [rsp+B8h] [rbp-B8h]
  __int64 v63; // [rsp+C0h] [rbp-B0h]
  __int64 v64; // [rsp+C8h] [rbp-A8h]
  __int16 v65; // [rsp+D0h] [rbp-A0h]
  __int64 v66; // [rsp+D8h] [rbp-98h]
  char *v67; // [rsp+E0h] [rbp-90h]
  __int64 v68; // [rsp+E8h] [rbp-88h]
  int v69; // [rsp+F0h] [rbp-80h]
  char v70; // [rsp+F4h] [rbp-7Ch]
  char v71; // [rsp+F8h] [rbp-78h] BYREF

  if ( !sub_D47930(a2) || !sub_D47840(a2) )
    return 0;
  v7 = sub_AA5930(**(_QWORD **)(a2 + 32));
  v46 = v8;
  v9 = v7;
  if ( v7 == v8 )
    return 1;
  while ( 1 )
  {
    v10 = *(_QWORD *)(a1 + 16);
    v50[0] = 6;
    v55 = v57;
    v50[1] = 0;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v56 = 0x200000000LL;
    v13 = sub_10238A0(v9, a2, v10, (__int64)v50, 0, 0);
    if ( (_BYTE)v13 )
    {
      v38 = *(unsigned int *)(a3 + 8);
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v38 + 1, 8u, v11, v12);
        v38 = *(unsigned int *)(a3 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v38) = v9;
      ++*(_DWORD *)(a3 + 8);
      goto LABEL_47;
    }
    if ( !a4 )
    {
      if ( *(_BYTE *)(a1 + 60) )
      {
        v39 = *(_QWORD **)(a1 + 40);
        v40 = &v39[*(unsigned int *)(a1 + 52)];
        if ( v39 == v40 )
          break;
        while ( *v39 != v9 )
        {
          if ( v40 == ++v39 )
            goto LABEL_64;
        }
      }
      else
      {
        v41 = sub_C8CA60(a1 + 32, v9);
        v13 = 0;
        if ( !v41 )
          break;
      }
      goto LABEL_47;
    }
    v14 = sub_D47930(a2);
    v15 = *(_QWORD *)(v9 - 8);
    v13 = 0;
    v16 = v14;
    if ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) != 0 )
    {
      v17 = 0;
      while ( v16 != *(_QWORD *)(v15 + 32LL * *(unsigned int *)(v9 + 72) + 8 * v17) )
      {
        if ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) == (_DWORD)++v17 )
          goto LABEL_69;
      }
      v18 = 32 * v17;
    }
    else
    {
LABEL_69:
      v18 = 0x1FFFFFFFE0LL;
    }
    v19 = *(unsigned __int8 **)(v15 + v18);
    while ( 1 )
    {
      v20 = *v19;
      if ( *v19 <= 0x1Cu )
        break;
      if ( v20 != 84 || (*((_DWORD *)v19 + 1) & 0x7FFFFFF) != 1 )
        goto LABEL_14;
      v19 = (unsigned __int8 *)**((_QWORD **)v19 - 1);
      if ( !v19 )
        BUG();
    }
    if ( v20 <= 0x15u )
      break;
LABEL_14:
    v21 = *((_QWORD *)v19 + 2);
    if ( !v21 )
      break;
    while ( 1 )
    {
      v22 = *(_QWORD *)(v21 + 24);
      if ( *(_BYTE *)v22 == 84 && (*(_DWORD *)(v22 + 4) & 0x7FFFFFF) != 1 )
        break;
      v21 = *(_QWORD *)(v21 + 8);
      if ( !v21 )
        goto LABEL_64;
    }
    v65 = 0;
    v42 = v22;
    v58 = 0;
    v59[0] = 6;
    v59[1] = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v66 = 0;
    v67 = &v71;
    v68 = 8;
    v69 = 0;
    v70 = 1;
    v23 = sub_1026850(v22, a4, (__int64)&v58, 0, 0, 0, 0);
    v26 = 0;
    v13 = 0;
    if ( v23 )
    {
      v26 = v42;
      if ( v63 )
        v26 = 0;
    }
    if ( !v70 )
    {
      v45 = v26;
      _libc_free((unsigned __int64)v67);
      v13 = 0;
      v26 = v45;
    }
    if ( v60 != -4096 && v60 != 0 && v60 != -8192 )
    {
      v44 = v26;
      sub_BD60C0(v59);
      v13 = 0;
      v26 = v44;
    }
    if ( !v26 )
      break;
    v27 = 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
    {
      v28 = *(char **)(v26 - 8);
      v29 = &v28[v27];
    }
    else
    {
      v29 = (char *)v26;
      v28 = (char *)(v26 - v27);
    }
    v30 = v27 >> 5;
    v31 = v27 >> 7;
    if ( v31 )
    {
      v31 = (__int64)&v28[128 * v31];
      while ( *(_QWORD *)v28 != v9 )
      {
        if ( *((_QWORD *)v28 + 4) == v9 )
        {
          v28 += 32;
          goto LABEL_37;
        }
        if ( *((_QWORD *)v28 + 8) == v9 )
        {
          v28 += 64;
          goto LABEL_37;
        }
        if ( *((_QWORD *)v28 + 12) == v9 )
        {
          v28 += 96;
          goto LABEL_37;
        }
        v28 += 128;
        if ( (char *)v31 == v28 )
        {
          v30 = (v29 - v28) >> 5;
          goto LABEL_86;
        }
      }
      goto LABEL_37;
    }
LABEL_86:
    if ( v30 == 2 )
      goto LABEL_90;
    if ( v30 != 3 )
    {
      if ( v30 != 1 )
        break;
LABEL_96:
      if ( *(_QWORD *)v28 != v9 )
        break;
      goto LABEL_37;
    }
    if ( *(_QWORD *)v28 != v9 )
    {
      v28 += 32;
LABEL_90:
      if ( *(_QWORD *)v28 != v9 )
      {
        v28 += 32;
        goto LABEL_96;
      }
    }
LABEL_37:
    if ( v29 == v28 )
      break;
    v32 = a1 + 32;
    if ( !*(_BYTE *)(a1 + 60) )
      goto LABEL_71;
    v31 = *(unsigned int *)(a1 + 52);
    v33 = *(char **)(a1 + 40);
    v29 = &v33[8 * v31];
    v34 = *(_DWORD *)(a1 + 52);
    if ( v33 != v29 )
    {
      v31 = *(_QWORD *)(a1 + 40);
      while ( v9 != *(_QWORD *)v31 )
      {
        v31 += 8;
        if ( v29 == (char *)v31 )
          goto LABEL_83;
      }
      v35 = (__int64)&v33[8 * v34];
      if ( (char *)v35 != v33 )
        goto LABEL_46;
      goto LABEL_74;
    }
LABEL_83:
    if ( v34 < *(_DWORD *)(a1 + 48) )
    {
      *(_DWORD *)(a1 + 52) = v34 + 1;
      *(_QWORD *)v29 = v9;
      v33 = *(char **)(a1 + 40);
      ++*(_QWORD *)(a1 + 32);
      v35 = *(unsigned __int8 *)(a1 + 60);
    }
    else
    {
LABEL_71:
      v43 = v26;
      sub_C8CC70(v32, v9, v31, (__int64)v29, v24, v25);
      v35 = *(unsigned __int8 *)(a1 + 60);
      v33 = *(char **)(a1 + 40);
      v26 = v43;
      v32 = a1 + 32;
    }
    if ( !(_BYTE)v35 )
      goto LABEL_76;
    v34 = *(_DWORD *)(a1 + 52);
    v35 = (__int64)&v33[8 * v34];
    if ( (char *)v35 != v33 )
    {
LABEL_46:
      while ( v26 != *(_QWORD *)v33 )
      {
        v33 += 8;
        if ( (char *)v35 == v33 )
          goto LABEL_74;
      }
      goto LABEL_47;
    }
LABEL_74:
    if ( v34 >= *(_DWORD *)(a1 + 48) )
    {
LABEL_76:
      sub_C8CC70(v32, v26, v35, (__int64)v29, v24, v25);
      goto LABEL_47;
    }
    *(_DWORD *)(a1 + 52) = v34 + 1;
    *(_QWORD *)v35 = v26;
    ++*(_QWORD *)(a1 + 32);
LABEL_47:
    if ( v55 != v57 )
      _libc_free((unsigned __int64)v55);
    if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
      sub_BD60C0(v50);
    if ( !v9 )
      BUG();
    v36 = *(_QWORD *)(v9 + 32);
    if ( !v36 )
      BUG();
    v9 = 0;
    if ( *(_BYTE *)(v36 - 24) == 84 )
      v9 = v36 - 24;
    if ( v46 == v9 )
      return 1;
  }
LABEL_64:
  if ( v55 != v57 )
  {
    v48 = v13;
    _libc_free((unsigned __int64)v55);
    v13 = v48;
  }
  if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
  {
    v49 = v13;
    sub_BD60C0(v50);
    return v49;
  }
  return v13;
}
