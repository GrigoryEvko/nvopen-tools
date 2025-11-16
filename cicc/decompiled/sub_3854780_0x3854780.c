// Function: sub_3854780
// Address: 0x3854780
//
__int64 __fastcall sub_3854780(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v5; // edx
  __int64 v6; // r9
  int v7; // edx
  int v8; // r8d
  __int64 v9; // rdi
  int v10; // ecx
  __int64 v11; // rsi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // esi
  unsigned int v20; // edi
  __int64 *v21; // rcx
  __int64 v22; // r15
  __int64 v23; // r8
  unsigned int v24; // edx
  __int64 v25; // rcx
  int v26; // esi
  __int64 *v27; // rax
  __int64 v28; // r11
  __int64 v29; // rax
  _QWORD *v30; // rdx
  int v32; // eax
  __int64 *v33; // rax
  bool v34; // cc
  int v35; // eax
  int v36; // r10d
  int v37; // ecx
  int v38; // r11d
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  unsigned __int64 v43; // rsi
  unsigned int v44; // edx
  unsigned __int64 v45; // rdi
  int v46; // eax
  int v47; // r10d
  __int64 v48; // rax
  __int64 v49; // [rsp+0h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-90h] BYREF
  __int64 v51; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v52[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v53; // [rsp+30h] [rbp-70h]
  __int64 v54; // [rsp+38h] [rbp-68h]
  _QWORD *v55; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v56; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v57[10]; // [rsp+50h] [rbp-50h] BYREF

  LODWORD(v2) = 0;
  v5 = *(_DWORD *)(a1 + 184);
  v52[0] = 0;
  v52[1] = -1;
  v53 = 0;
  v54 = 0;
  if ( v5 && *(_DWORD *)(a1 + 216) )
  {
    LOBYTE(v32) = sub_384F1D0(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), &v50, v52);
    LODWORD(v2) = v32;
  }
  if ( !sub_15FA300(a2) )
  {
LABEL_10:
    v8 = *(_DWORD *)(a2 + 20);
    goto LABEL_11;
  }
  v7 = *(_DWORD *)(a1 + 256);
  v8 = *(_DWORD *)(a2 + 20);
  if ( v7 )
  {
    v9 = *(_QWORD *)(a1 + 240);
    v10 = v7 - 1;
    v11 = *(_QWORD *)(a2 - 24LL * (v8 & 0xFFFFFFF));
    v12 = (v7 - 1) & (((unsigned int)*(_QWORD *)(a2 - 24LL * (v8 & 0xFFFFFFF)) >> 9) ^ ((unsigned int)v11 >> 4));
    v13 = (__int64 *)(v9 + 32LL * v12);
    v6 = *v13;
    if ( *v13 != v11 )
    {
      v46 = 1;
      while ( v6 != -8 )
      {
        v47 = v46 + 1;
        v12 = v10 & (v46 + v12);
        v13 = (__int64 *)(v9 + 32LL * v12);
        v6 = *v13;
        if ( v11 == *v13 )
          goto LABEL_7;
        v46 = v47;
      }
      goto LABEL_11;
    }
LABEL_7:
    v14 = v13[1];
    v55 = (_QWORD *)v14;
    LODWORD(v57[0]) = *((_DWORD *)v13 + 6);
    if ( LODWORD(v57[0]) > 0x40 )
    {
      sub_16A4FD0((__int64)&v56, (const void **)v13 + 2);
      if ( !v55 )
        goto LABEL_55;
      goto LABEL_36;
    }
    v56 = v13[2];
    if ( v14 )
    {
LABEL_36:
      if ( (unsigned __int8)sub_384FD60(a1, a2, (__int64)&v56) )
      {
        v51 = a2;
        v33 = sub_3854530(a1 + 232, &v51);
        v34 = *((_DWORD *)v33 + 6) <= 0x40u;
        v33[1] = (__int64)v55;
        if ( v34 && LODWORD(v57[0]) <= 0x40 )
        {
          v43 = v56;
          v33[2] = v56;
          v44 = v57[0];
          *((_DWORD *)v33 + 6) = v57[0];
          v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
          if ( v44 > 0x40 )
          {
            v48 = (unsigned int)(((unsigned __int64)v44 + 63) >> 6) - 1;
            *(_QWORD *)(v43 + 8 * v48) &= v45;
          }
          else
          {
            v33[2] = v45 & v43;
          }
        }
        else
        {
          sub_16A51C0((__int64)(v33 + 2), (__int64)&v56);
        }
        if ( LODWORD(v57[0]) > 0x40 && v56 )
          j_j___libc_free_0_0(v56);
        goto LABEL_43;
      }
LABEL_55:
      if ( LODWORD(v57[0]) > 0x40 && v56 )
      {
        j_j___libc_free_0_0(v56);
        v8 = *(_DWORD *)(a2 + 20);
        goto LABEL_11;
      }
      goto LABEL_10;
    }
  }
LABEL_11:
  v15 = v8 & 0xFFFFFFF;
  v16 = (__int64 *)(a2 + 24 * (1 - v15));
  if ( (__int64 *)a2 == v16 )
  {
LABEL_43:
    if ( (_BYTE)v2 )
    {
      v55 = (_QWORD *)a2;
      v42 = sub_176FB00(a1 + 168, (__int64 *)&v55);
      v42[1] = v50;
    }
    else
    {
      LODWORD(v2) = 1;
    }
    return (unsigned int)v2;
  }
  v17 = a2 + 24 * (1 - v15);
  while ( 1 )
  {
    v18 = *(_QWORD *)v17;
    if ( *(_BYTE *)(*(_QWORD *)v17 + 16LL) <= 0x10u )
      goto LABEL_13;
    v19 = *(_DWORD *)(a1 + 160);
    if ( !v19 )
      goto LABEL_18;
    v6 = *(_QWORD *)(a1 + 144);
    v20 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v21 = (__int64 *)(v6 + 16LL * v20);
    v22 = *v21;
    if ( v18 != *v21 )
      break;
LABEL_17:
    if ( !v21[1] )
      goto LABEL_18;
LABEL_13:
    v17 += 24;
    if ( a2 == v17 )
      goto LABEL_43;
  }
  v37 = 1;
  while ( v22 != -8 )
  {
    v38 = v37 + 1;
    v20 = (v19 - 1) & (v37 + v20);
    v21 = (__int64 *)(v6 + 16LL * v20);
    v22 = *v21;
    if ( v18 == *v21 )
      goto LABEL_17;
    v37 = v38;
  }
LABEL_18:
  if ( (_BYTE)v2 )
  {
    sub_384F170(a1, v53);
    v39 = *(_DWORD *)(a2 + 20);
    v55 = v57;
    v40 = v39 & 0xFFFFFFF;
    v16 = (__int64 *)(a2 + 24 * (v41 - v40));
    v57[0] = *(_QWORD *)(a2 - 24 * v40);
    v56 = 0x400000001LL;
    if ( (__int64 *)a2 == v16 )
    {
      v25 = 1;
      v30 = v57;
      goto LABEL_32;
    }
    v19 = *(_DWORD *)(a1 + 160);
  }
  else
  {
    v23 = -3 * v15;
    v55 = v57;
    v57[0] = *(_QWORD *)(a2 + 8 * v23);
    v56 = 0x400000001LL;
  }
  v24 = 4;
  v25 = 1;
  while ( 2 )
  {
    v2 = *v16;
    if ( !v19 )
      goto LABEL_28;
    v26 = v19 - 1;
    v23 = *(_QWORD *)(a1 + 144);
    LODWORD(v6) = v26 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v27 = (__int64 *)(v23 + 16LL * (unsigned int)v6);
    v28 = *v27;
    if ( *v27 != v2 )
    {
      v35 = 1;
      while ( v28 != -8 )
      {
        v36 = v35 + 1;
        LODWORD(v6) = v26 & (v35 + v6);
        v27 = (__int64 *)(v23 + 16LL * (unsigned int)v6);
        v28 = *v27;
        if ( v2 == *v27 )
          goto LABEL_22;
        v35 = v36;
      }
      goto LABEL_28;
    }
LABEL_22:
    v29 = v27[1];
    if ( !v29 )
    {
LABEL_28:
      if ( v24 <= (unsigned int)v25 )
      {
        sub_16CD150((__int64)&v55, v57, 0, 8, v23, v6);
        v25 = (unsigned int)v56;
      }
      v16 += 3;
      v55[v25] = v2;
      v25 = (unsigned int)(v56 + 1);
      LODWORD(v56) = v56 + 1;
      if ( (__int64 *)a2 == v16 )
        break;
      goto LABEL_26;
    }
    if ( v24 <= (unsigned int)v25 )
    {
      v49 = v29;
      sub_16CD150((__int64)&v55, v57, 0, 8, v23, v6);
      v25 = (unsigned int)v56;
      v29 = v49;
    }
    v16 += 3;
    v55[v25] = v29;
    v25 = (unsigned int)(v56 + 1);
    LODWORD(v56) = v56 + 1;
    if ( (__int64 *)a2 != v16 )
    {
LABEL_26:
      v24 = HIDWORD(v56);
      v19 = *(_DWORD *)(a1 + 160);
      continue;
    }
    break;
  }
  v30 = v55;
LABEL_32:
  LOBYTE(v2) = (unsigned int)sub_14A5330(*(__int64 ***)a1, a2, (__int64)v30, v25) == 0;
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  return (unsigned int)v2;
}
