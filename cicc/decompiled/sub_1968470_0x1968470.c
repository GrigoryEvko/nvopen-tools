// Function: sub_1968470
// Address: 0x1968470
//
__int64 __fastcall sub_1968470(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 *v16; // rdi
  __int64 *v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rsi
  _QWORD *v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rbx
  __int64 v30; // rax
  char v31; // di
  unsigned int v32; // edx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // r14
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  _QWORD *v40; // r9
  __int64 v41; // rax
  __int64 v42; // rcx
  unsigned __int64 v43; // r8
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  char v53; // bl
  __int64 v54; // r13
  __int64 v55; // r15
  __int64 v56; // rbx
  __int64 v57; // r14
  char v58; // bl
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  int v66; // eax
  unsigned int v67; // eax
  unsigned int v68; // edx
  _QWORD *v69; // r13
  _QWORD *i; // r14
  unsigned __int64 v71; // rdi
  __int64 v72; // [rsp+10h] [rbp-A0h]
  char v73; // [rsp+1Fh] [rbp-91h]
  __int64 v77; // [rsp+38h] [rbp-78h]
  unsigned int v78; // [rsp+38h] [rbp-78h]
  unsigned __int8 v79; // [rsp+4Fh] [rbp-61h] BYREF
  _BYTE *v80; // [rsp+50h] [rbp-60h] BYREF
  __int64 v81; // [rsp+58h] [rbp-58h]
  _BYTE v82[80]; // [rsp+60h] [rbp-50h] BYREF

  v77 = sub_13FC520((__int64)a1);
  if ( !v77 )
    return 0;
  v73 = sub_13FC370((__int64)a1);
  if ( !v73 || a1[2] != a1[1] )
    return 0;
  v6 = sub_13FA560((__int64)a1);
  if ( !v6 )
  {
    v80 = v82;
    v81 = 0x400000000LL;
    sub_13F9CA0((__int64)a1, (__int64)&v80);
    result = 0;
    goto LABEL_99;
  }
  v7 = sub_13FC520((__int64)a1);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 80LL);
  if ( !v8 || v7 != v8 - 24 )
  {
    v9 = *(_QWORD *)(v7 + 8);
    if ( !v9 )
    {
LABEL_12:
      v11 = sub_157F280(v6);
      v13 = v12;
      v14 = v11;
      while ( v13 != v14 )
      {
        v15 = sub_1599EF0(*(__int64 ***)v14);
        if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
        {
          v17 = *(__int64 **)(v14 - 8);
          v16 = &v17[3 * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)];
        }
        else
        {
          v16 = (__int64 *)v14;
          v17 = (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        }
        for ( ; v17 != v16; v17 += 3 )
        {
          if ( *v17 )
          {
            v18 = v17[1];
            v19 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v19 = v18;
            if ( v18 )
              *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
          }
          *v17 = v15;
          if ( v15 )
          {
            v20 = *(_QWORD *)(v15 + 8);
            v17[1] = v20;
            if ( v20 )
              *(_QWORD *)(v20 + 16) = (unsigned __int64)(v17 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
            v17[2] = (v15 + 8) | v17[2] & 3;
            *(_QWORD *)(v15 + 8) = v17;
          }
        }
        v21 = *(_QWORD *)(v14 + 32);
        if ( !v21 )
          BUG();
        v14 = 0;
        if ( *(_BYTE *)(v21 - 8) == 77 )
          v14 = v21 - 24;
      }
      sub_1B17C80(a1, a2, a3, a4);
      return 2;
    }
    while ( 1 )
    {
      v10 = sub_1648700(v9);
      if ( (unsigned __int8)(*((_BYTE *)v10 + 16) - 25) <= 9u )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_12;
    }
LABEL_36:
    v26 = sub_157EBA0(v10[5]);
    if ( *(_BYTE *)(v26 + 16) == 26 && (*(_DWORD *)(v26 + 20) & 0xFFFFFFF) == 3 )
    {
      v22 = *(_QWORD *)(v26 - 72);
      if ( *(_BYTE *)(v22 + 16) == 13 )
      {
        v23 = *(_QWORD *)(v26 - 24);
        v24 = *(_QWORD *)(v26 - 48);
        v25 = *(_QWORD **)(v22 + 24);
        if ( *(_DWORD *)(v22 + 32) > 0x40u )
          v25 = (_QWORD *)*v25;
        if ( !v25 )
          v23 = v24;
        if ( v7 != v23 )
        {
          while ( 1 )
          {
            v9 = *(_QWORD *)(v9 + 8);
            if ( !v9 )
              goto LABEL_12;
            v10 = sub_1648700(v9);
            if ( (unsigned __int8)(*((_BYTE *)v10 + 16) - 25) <= 9u )
              goto LABEL_36;
          }
        }
      }
    }
  }
  v80 = v82;
  v81 = 0x400000000LL;
  sub_13F9CA0((__int64)a1, (__int64)&v80);
  v79 = 0;
  v27 = sub_157F280(v6);
  v72 = v28;
  v29 = v27;
  if ( v27 == v28 )
    goto LABEL_79;
  do
  {
    v30 = 0x17FFFFFFE8LL;
    v31 = *(_BYTE *)(v29 + 23) & 0x40;
    v32 = *(_DWORD *)(v29 + 20) & 0xFFFFFFF;
    if ( v32 )
    {
      v33 = 24LL * *(unsigned int *)(v29 + 56) + 8;
      v34 = 0;
      do
      {
        v35 = v29 - 24LL * v32;
        if ( v31 )
          v35 = *(_QWORD *)(v29 - 8);
        if ( *(_QWORD *)v80 == *(_QWORD *)(v35 + v33) )
        {
          v30 = 24 * v34;
          goto LABEL_46;
        }
        ++v34;
        v33 += 8;
      }
      while ( v32 != (_DWORD)v34 );
      v30 = 0x17FFFFFFE8LL;
    }
LABEL_46:
    if ( v31 )
      v36 = *(_QWORD *)(v29 - 8);
    else
      v36 = v29 - 24LL * v32;
    v37 = *(_QWORD *)(v36 + v30);
    v38 = v80 + 8;
    v39 = 8LL * (unsigned int)v81 - 8;
    v40 = &v80[8 * (unsigned int)v81];
    v41 = v39 >> 5;
    v42 = v39 >> 3;
    if ( v41 <= 0 )
    {
LABEL_70:
      if ( v42 != 2 )
      {
        if ( v42 != 3 )
        {
          if ( v42 != 1 )
            goto LABEL_73;
          goto LABEL_124;
        }
        v59 = 0x17FFFFFFE8LL;
        if ( v32 )
        {
          v60 = 0;
          do
          {
            if ( *v38 == *(_QWORD *)(v36 + 24LL * *(unsigned int *)(v29 + 56) + 8 * v60 + 8) )
            {
              v59 = 24 * v60;
              goto LABEL_115;
            }
            ++v60;
          }
          while ( v32 != (_DWORD)v60 );
          v59 = 0x17FFFFFFE8LL;
        }
LABEL_115:
        if ( v37 != *(_QWORD *)(v36 + v59) )
          goto LABEL_96;
        ++v38;
      }
      v61 = 0x17FFFFFFE8LL;
      if ( v32 )
      {
        v62 = 0;
        do
        {
          if ( *v38 == *(_QWORD *)(v36 + 24LL * *(unsigned int *)(v29 + 56) + 8 * v62 + 8) )
          {
            v61 = 24 * v62;
            goto LABEL_122;
          }
          ++v62;
        }
        while ( v32 != (_DWORD)v62 );
        v61 = 0x17FFFFFFE8LL;
      }
LABEL_122:
      if ( v37 != *(_QWORD *)(v36 + v61) )
        goto LABEL_96;
      ++v38;
LABEL_124:
      v63 = 0x17FFFFFFE8LL;
      if ( v32 )
      {
        v64 = 0;
        do
        {
          if ( *v38 == *(_QWORD *)(v36 + 24LL * *(unsigned int *)(v29 + 56) + 8 * v64 + 8) )
          {
            v63 = 24 * v64;
            goto LABEL_129;
          }
          ++v64;
        }
        while ( v32 != (_DWORD)v64 );
        v63 = 0x17FFFFFFE8LL;
      }
LABEL_129:
      if ( v37 != *(_QWORD *)(v36 + v63) )
        goto LABEL_96;
      goto LABEL_73;
    }
    v43 = (unsigned __int64)&v80[32 * v41 + 8];
    while ( 1 )
    {
      if ( v32 )
      {
        v44 = *(unsigned int *)(v29 + 56);
        v45 = 0;
        do
        {
          if ( *v38 == *(_QWORD *)(v36 + 24 * v44 + 8 * v45 + 8) )
          {
            v46 = 24 * v45;
            goto LABEL_55;
          }
          ++v45;
        }
        while ( v32 != (_DWORD)v45 );
        v46 = 0x17FFFFFFE8LL;
LABEL_55:
        if ( v37 != *(_QWORD *)(v36 + v46) )
          goto LABEL_96;
        v47 = 0;
        do
        {
          if ( *(_QWORD *)(v36 + 24 * v44 + 8 * v47 + 8) == v38[1] )
          {
            if ( v37 != *(_QWORD *)(v36 + 24 * v47) )
              goto LABEL_103;
            goto LABEL_60;
          }
          ++v47;
        }
        while ( v32 != (_DWORD)v47 );
        if ( v37 != *(_QWORD *)(v36 + 0x17FFFFFFE8LL) )
        {
LABEL_103:
          if ( v40 != v38 + 1 )
            goto LABEL_97;
          goto LABEL_73;
        }
LABEL_60:
        v48 = 0;
        do
        {
          if ( *(_QWORD *)(v36 + 24 * v44 + 8 * v48 + 8) == v38[2] )
          {
            if ( v37 != *(_QWORD *)(v36 + 24 * v48) )
              goto LABEL_106;
            goto LABEL_64;
          }
          ++v48;
        }
        while ( v32 != (_DWORD)v48 );
        if ( v37 != *(_QWORD *)(v36 + 0x17FFFFFFE8LL) )
        {
LABEL_106:
          if ( v40 != v38 + 2 )
          {
LABEL_97:
            v58 = 0;
            goto LABEL_98;
          }
          goto LABEL_73;
        }
LABEL_64:
        v49 = 0;
        v50 = v36 + 24 * v44;
        while ( v38[3] != *(_QWORD *)(v50 + 8 * v49 + 8) )
        {
          if ( v32 == (_DWORD)++v49 )
            goto LABEL_94;
        }
        if ( v37 != *(_QWORD *)(v36 + 24 * v49) )
          break;
        goto LABEL_68;
      }
      if ( v37 != *(_QWORD *)(v36 + 0x17FFFFFFE8LL) )
        goto LABEL_96;
LABEL_94:
      if ( v37 != *(_QWORD *)(v36 + 0x17FFFFFFE8LL) )
        break;
LABEL_68:
      v38 += 4;
      if ( (_QWORD *)v43 == v38 )
      {
        v42 = v40 - v38;
        goto LABEL_70;
      }
    }
    v38 += 3;
LABEL_96:
    if ( v40 != v38 )
      goto LABEL_97;
LABEL_73:
    if ( *(_BYTE *)(v37 + 16) > 0x17u )
    {
      v51 = sub_157EBA0(v77);
      if ( !(unsigned __int8)sub_13FC570((__int64)a1, v37, &v79, v51) )
      {
        v58 = v73;
        v73 = 0;
LABEL_98:
        result = v79;
        if ( v79 )
        {
          v53 = v73 & v58;
          goto LABEL_138;
        }
        goto LABEL_99;
      }
    }
    v52 = *(_QWORD *)(v29 + 32);
    if ( !v52 )
      BUG();
    v29 = 0;
    if ( *(_BYTE *)(v52 - 8) == 77 )
      v29 = v52 - 24;
  }
  while ( v72 != v29 );
LABEL_79:
  v53 = v79;
  if ( !v79 )
    goto LABEL_80;
LABEL_138:
  v66 = *(_DWORD *)(a3 + 672);
  ++*(_QWORD *)(a3 + 656);
  if ( v66 || *(_DWORD *)(a3 + 676) )
  {
    v67 = 4 * v66;
    v68 = *(_DWORD *)(a3 + 680);
    if ( v67 < 0x40 )
      v67 = 64;
    if ( v67 < v68 )
    {
      sub_195EB50(a3 + 656);
    }
    else
    {
      v69 = *(_QWORD **)(a3 + 664);
      for ( i = &v69[5 * v68]; i != v69; v69 += 5 )
      {
        if ( *v69 != -8 )
        {
          if ( *v69 != -16 )
          {
            v71 = v69[1];
            if ( (_QWORD *)v71 != v69 + 3 )
              _libc_free(v71);
          }
          *v69 = -8;
        }
      }
      *(_QWORD *)(a3 + 672) = 0;
    }
  }
  if ( !v53 )
  {
LABEL_92:
    result = v79;
    goto LABEL_99;
  }
LABEL_80:
  v54 = a1[4];
  v55 = a1[5];
  if ( v54 != v55 )
  {
    while ( 2 )
    {
      v56 = *(_QWORD *)(*(_QWORD *)v54 + 48LL);
      v57 = *(_QWORD *)v54 + 40LL;
      if ( v57 != v56 )
      {
        if ( !v56 )
          goto LABEL_90;
LABEL_86:
        if ( (unsigned __int8)sub_15F3040(v56 - 24) || sub_15F3330(v56 - 24) )
        {
          if ( v57 != v56 )
            goto LABEL_92;
        }
        else
        {
          while ( 1 )
          {
            v56 = *(_QWORD *)(v56 + 8);
            if ( v57 == v56 )
              break;
            if ( v56 )
              goto LABEL_86;
LABEL_90:
            if ( (unsigned __int8)sub_15F3040(0) || sub_15F3330(0) )
              goto LABEL_92;
          }
        }
      }
      v54 += 8;
      if ( v55 == v54 )
        break;
      continue;
    }
  }
  v65 = sub_1474260(a3, (__int64)a1);
  if ( sub_14562D0(v65) )
    goto LABEL_92;
  sub_1B17C80(a1, a2, a3, a4);
  result = 2;
LABEL_99:
  if ( v80 != v82 )
  {
    v78 = result;
    _libc_free((unsigned __int64)v80);
    return v78;
  }
  return result;
}
