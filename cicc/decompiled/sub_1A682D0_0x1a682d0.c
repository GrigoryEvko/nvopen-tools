// Function: sub_1A682D0
// Address: 0x1a682d0
//
__int64 __fastcall sub_1A682D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r13
  __int64 *v6; // r14
  int v7; // eax
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // r10
  char *v11; // rbx
  char *v12; // r10
  __int64 v13; // r10
  __int64 **v14; // r11
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r12
  _BYTE *v17; // rdx
  int v18; // esi
  _BYTE *v19; // r8
  __int64 v20; // rcx
  char v21; // al
  __int64 *v22; // rbx
  _BYTE *v23; // rsi
  _BYTE *v24; // r9
  unsigned __int64 v25; // rdi
  int v26; // eax
  int v27; // r10d
  unsigned int v28; // r12d
  _QWORD *v29; // rcx
  _QWORD *v30; // rax
  __int64 v31; // r14
  _QWORD *v32; // r12
  __int64 v33; // r15
  _QWORD *v34; // r15
  unsigned __int64 v35; // rax
  __int64 *v37; // rsi
  __int64 *v38; // rax
  __int64 *v39; // rcx
  __int64 v40; // rax
  __int64 *v41; // r9
  __int64 v42; // r13
  __int64 *v43; // r15
  __int64 v44; // r12
  _QWORD *v45; // rax
  _BYTE *v46; // r14
  _BYTE *v47; // rdx
  _QWORD *v48; // rdx
  __int64 v51; // [rsp+18h] [rbp-108h]
  unsigned int v53; // [rsp+2Ch] [rbp-F4h]
  int v54; // [rsp+30h] [rbp-F0h]
  __int64 **v55; // [rsp+30h] [rbp-F0h]
  __int64 v56; // [rsp+38h] [rbp-E8h]
  __int64 v57; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v58; // [rsp+40h] [rbp-E0h]
  __int64 v59; // [rsp+48h] [rbp-D8h]
  _BYTE *v60; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+58h] [rbp-C8h]
  _BYTE v62[32]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v64; // [rsp+88h] [rbp-98h]
  _BYTE *v65; // [rsp+90h] [rbp-90h]
  __int64 v66; // [rsp+98h] [rbp-88h]
  int v67; // [rsp+A0h] [rbp-80h]
  _BYTE v68[120]; // [rsp+A8h] [rbp-78h] BYREF

  v64 = v68;
  v5 = *(_QWORD *)(a2 + 48);
  v65 = v68;
  v63 = 0;
  v66 = 8;
  v67 = 0;
  v56 = a2 + 40;
  if ( v5 == a2 + 40 )
    return 0;
  v53 = 0;
  v6 = &v63;
  do
  {
    if ( !v5 )
      BUG();
    v7 = *(unsigned __int8 *)(v5 - 8);
    v8 = v5 - 24;
    if ( (unsigned __int8)v7 <= 0x17u )
    {
      if ( (_BYTE)v7 != 5 )
      {
LABEL_47:
        v23 = v65;
        v24 = v64;
        v25 = (unsigned __int64)v65;
        if ( v65 == v64 )
          goto LABEL_48;
LABEL_19:
        sub_16CCBA0((__int64)v6, v8);
        v26 = v67;
        v25 = (unsigned __int64)v65;
        v24 = v64;
        v27 = HIDWORD(v66);
        goto LABEL_20;
      }
      v9 = *(unsigned __int16 *)(v5 - 6);
    }
    else
    {
      v9 = v7 - 24;
    }
    switch ( v9 )
    {
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 19:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 32:
      case 37:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 51:
      case 52:
      case 54:
      case 55:
        v10 = 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
        {
          v11 = *(char **)(v5 - 32);
          v12 = &v11[v10];
        }
        else
        {
          v11 = (char *)(v8 - v10);
          v12 = (char *)(v5 - 24);
        }
        v13 = v12 - v11;
        v60 = v62;
        v14 = *(__int64 ***)(a1 + 8);
        v61 = 0x400000000LL;
        v15 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
        v16 = v15;
        if ( (unsigned __int64)v13 > 0x60 )
        {
          v51 = v13;
          v55 = v14;
          v58 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
          sub_16CD150((__int64)&v60, v62, v58, 8, a5, (int)v62);
          v19 = v60;
          v18 = v61;
          LODWORD(v15) = v58;
          v14 = v55;
          v13 = v51;
          v17 = &v60[8 * (unsigned int)v61];
        }
        else
        {
          v17 = v62;
          v18 = 0;
          v19 = v62;
        }
        if ( v13 > 0 )
        {
          do
          {
            v20 = *(_QWORD *)v11;
            v17 += 8;
            v11 += 24;
            *((_QWORD *)v17 - 1) = v20;
            --v16;
          }
          while ( v16 );
          v19 = v60;
          v18 = v61;
        }
        LODWORD(v61) = v18 + v15;
        v54 = sub_14A5330(v14, v5 - 24, (__int64)v19, (unsigned int)(v18 + v15));
        if ( v60 != v62 )
          _libc_free((unsigned __int64)v60);
        if ( v54 == -1 )
          goto LABEL_47;
        v21 = sub_14AF470(v5 - 24, 0, 0, 0);
        v22 = (__int64 *)(v5 - 24);
        v23 = v65;
        if ( !v21 )
          goto LABEL_18;
        v40 = 3LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
        v41 = (__int64 *)(v8 - v40 * 8);
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
        {
          v41 = *(__int64 **)(v5 - 32);
          v22 = &v41[v40];
        }
        if ( v22 == v41 )
          goto LABEL_71;
        v59 = v5;
        v42 = (__int64)v6;
        v57 = v8;
        v43 = v41;
        break;
      default:
        goto LABEL_47;
    }
    while ( 1 )
    {
      v44 = *v43;
      if ( *(_BYTE *)(*v43 + 16) > 0x17u )
        break;
LABEL_69:
      v43 += 3;
      if ( v22 == v43 )
      {
        v6 = (__int64 *)v42;
        v5 = v59;
LABEL_71:
        v53 += v54;
        if ( v53 <= dword_4FB4D40 )
          goto LABEL_21;
        v25 = (unsigned __int64)v65;
        v24 = v64;
        v28 = 0;
        goto LABEL_44;
      }
    }
    v23 = v65;
    v45 = v64;
    if ( v65 == v64 )
    {
      v46 = &v65[8 * HIDWORD(v66)];
      if ( v65 == v46 )
      {
        v47 = v65;
      }
      else
      {
        do
        {
          if ( v44 == *v45 )
            break;
          ++v45;
        }
        while ( v46 != (_BYTE *)v45 );
        v47 = &v65[8 * HIDWORD(v66)];
      }
    }
    else
    {
      v46 = &v65[8 * (unsigned int)v66];
      v45 = sub_16CC9F0(v42, *v43);
      if ( v44 == *v45 )
      {
        v23 = v65;
        if ( v65 == v64 )
          v47 = &v65[8 * HIDWORD(v66)];
        else
          v47 = &v65[8 * (unsigned int)v66];
      }
      else
      {
        v23 = v65;
        if ( v65 != v64 )
        {
          v45 = &v65[8 * (unsigned int)v66];
          goto LABEL_67;
        }
        v45 = &v65[8 * HIDWORD(v66)];
        v47 = v45;
      }
    }
    for ( ; v47 != (_BYTE *)v45; ++v45 )
    {
      if ( *v45 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
LABEL_67:
    if ( v46 == (_BYTE *)v45 )
      goto LABEL_69;
    v6 = (__int64 *)v42;
    v8 = v57;
    v5 = v59;
LABEL_18:
    v24 = v64;
    v25 = (unsigned __int64)v23;
    if ( v23 != v64 )
      goto LABEL_19;
LABEL_48:
    v37 = (__int64 *)&v23[8 * HIDWORD(v66)];
    if ( (__int64 *)v25 == v37 )
      goto LABEL_89;
    v38 = (__int64 *)v25;
    v39 = 0;
    while ( 2 )
    {
      if ( v8 == *v38 )
      {
        if ( dword_4FB4C60 >= (unsigned int)(HIDWORD(v66) - v67) )
          goto LABEL_21;
        goto LABEL_55;
      }
      if ( *v38 == -2 )
        v39 = v38;
      if ( v37 != ++v38 )
        continue;
      break;
    }
    if ( v39 )
    {
      *v39 = v8;
      ++v63;
      v25 = (unsigned __int64)v65;
      v26 = v67 - 1;
      v24 = v64;
      v27 = HIDWORD(v66);
      --v67;
      goto LABEL_20;
    }
LABEL_89:
    if ( HIDWORD(v66) >= (unsigned int)v66 )
      goto LABEL_19;
    ++HIDWORD(v66);
    *v37 = v8;
    v24 = v64;
    ++v63;
    v27 = HIDWORD(v66);
    v26 = v67;
    v25 = (unsigned __int64)v65;
LABEL_20:
    if ( dword_4FB4C60 < (unsigned int)(v27 - v26) )
    {
LABEL_55:
      v28 = 0;
      goto LABEL_44;
    }
LABEL_21:
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v56 != v5 );
  v25 = (unsigned __int64)v65;
  v24 = v64;
  v28 = 0;
  v29 = v65;
  v30 = v64;
  if ( v53 )
  {
    v31 = *(_QWORD *)(a2 + 48);
    if ( v56 == v31 )
    {
LABEL_43:
      v28 = 1;
      goto LABEL_44;
    }
    while ( 2 )
    {
      v33 = v31;
      v31 = *(_QWORD *)(v31 + 8);
      v34 = (_QWORD *)(v33 - 24);
      if ( v29 == v30 )
      {
        v32 = &v29[HIDWORD(v66)];
        if ( v29 == v32 )
        {
          v48 = v29;
        }
        else
        {
          do
          {
            if ( v34 == (_QWORD *)*v30 )
              break;
            ++v30;
          }
          while ( v32 != v30 );
          v48 = &v29[HIDWORD(v66)];
        }
        break;
      }
      v32 = &v29[(unsigned int)v66];
      v30 = sub_16CC9F0((__int64)&v63, (__int64)v34);
      if ( v34 == (_QWORD *)*v30 )
      {
        v29 = v65;
        if ( v65 == v64 )
          v48 = &v65[8 * HIDWORD(v66)];
        else
          v48 = &v65[8 * (unsigned int)v66];
      }
      else
      {
        v29 = v65;
        if ( v65 != v64 )
        {
          v30 = &v65[8 * (unsigned int)v66];
LABEL_28:
          if ( v32 == v30 )
          {
LABEL_41:
            v35 = sub_157EBA0(a3);
            sub_15F22F0(v34, v35);
            v29 = v65;
            if ( v5 == v31 )
            {
LABEL_42:
              v24 = v64;
              v25 = (unsigned __int64)v29;
              goto LABEL_43;
            }
          }
          else
          {
LABEL_29:
            if ( v5 == v31 )
              goto LABEL_42;
          }
          v30 = v64;
          continue;
        }
        v30 = &v65[8 * HIDWORD(v66)];
        v48 = v30;
      }
      break;
    }
    if ( v30 != v48 )
    {
      while ( *v30 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v48 == ++v30 )
        {
          if ( v32 != v30 )
            goto LABEL_29;
          goto LABEL_41;
        }
      }
    }
    goto LABEL_28;
  }
LABEL_44:
  if ( v24 != (_BYTE *)v25 )
    _libc_free(v25);
  return v28;
}
