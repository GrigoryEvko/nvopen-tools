// Function: sub_2286A70
// Address: 0x2286a70
//
__int64 __fastcall sub_2286A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rbx
  _BYTE *v11; // rdi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // r14
  __int64 v20; // rdi
  __int64 *v21; // r15
  unsigned __int64 v22; // rbx
  unsigned int v23; // esi
  unsigned int v24; // r10d
  __int64 *v25; // rdi
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rax
  unsigned int v30; // r11d
  int v31; // ecx
  int v32; // edx
  int v33; // r8d
  int v34; // esi
  unsigned int v35; // r15d
  __int64 *v36; // rcx
  __int64 v37; // rdi
  int v38; // r8d
  unsigned int v39; // ecx
  __int64 v40; // r11
  int v41; // edi
  __int64 *v42; // rsi
  __int64 v43; // r15
  __int64 v44; // r13
  __int64 v45; // rcx
  __int64 v46; // rbx
  __int64 v47; // r14
  char v48; // al
  __int64 *v49; // rdx
  __int64 v50; // rsi
  __int64 *v51; // rax
  __int64 v52; // rbx
  __int64 v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rdi
  unsigned __int64 v56; // rdi
  char v57; // dl
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // [rsp+10h] [rbp-F0h]
  __int64 v62; // [rsp+18h] [rbp-E8h]
  __int64 v63; // [rsp+28h] [rbp-D8h]
  __int64 v64; // [rsp+28h] [rbp-D8h]
  __int64 v65; // [rsp+30h] [rbp-D0h] BYREF
  __int64 *v66; // [rsp+38h] [rbp-C8h]
  __int64 v67; // [rsp+40h] [rbp-C0h]
  int v68; // [rsp+48h] [rbp-B8h]
  char v69; // [rsp+4Ch] [rbp-B4h]
  _BYTE v70[176]; // [rsp+50h] [rbp-B0h] BYREF

  result = a2 + 24;
  *(_QWORD *)(a1 + 8) = a3;
  v61 = a1 + 16;
  *(_QWORD *)(a1 + 80) = sub_22857A0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 56) = a4;
  *(_QWORD *)(a1 + 64) = a5;
  *(_QWORD *)(a1 + 72) = sub_2285AF0;
  v8 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v63 = v8;
  v62 = a2 + 24;
  if ( a2 + 24 == v8 )
    goto LABEL_26;
  do
  {
    if ( !v63 )
    {
      v65 = 0;
      v67 = 16;
      v66 = (__int64 *)v70;
      v68 = 0;
      v69 = 1;
      BUG();
    }
    v9 = v63 - 56;
    v65 = 0;
    v67 = 16;
    v66 = (__int64 *)v70;
    v68 = 0;
    v69 = 1;
    v10 = *(_QWORD *)(v63 - 40);
    if ( !v10 )
    {
LABEL_18:
      v22 = 0;
      goto LABEL_19;
    }
    do
    {
      while ( 1 )
      {
        v11 = *(_BYTE **)(v10 + 24);
        if ( *v11 == 85 )
          break;
LABEL_5:
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_13;
      }
      v14 = sub_B43CB0((__int64)v11);
      if ( !v69 )
        goto LABEL_36;
      v17 = v66;
      v13 = HIDWORD(v67);
      v12 = &v66[HIDWORD(v67)];
      if ( v66 != v12 )
      {
        while ( v14 != *v17 )
        {
          if ( v12 == ++v17 )
            goto LABEL_11;
        }
        goto LABEL_5;
      }
LABEL_11:
      if ( HIDWORD(v67) >= (unsigned int)v67 )
      {
LABEL_36:
        sub_C8CC70((__int64)&v65, v14, (__int64)v12, v13, v15, v16);
        goto LABEL_5;
      }
      ++HIDWORD(v67);
      *v12 = v14;
      ++v65;
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( v10 );
LABEL_13:
    v18 = v66;
    if ( v69 )
      v19 = &v66[HIDWORD(v67)];
    else
      v19 = &v66[(unsigned int)v67];
    if ( v66 == v19 )
      goto LABEL_18;
    while ( 1 )
    {
      v20 = *v18;
      v21 = v18;
      if ( (unsigned __int64)*v18 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v19 == ++v18 )
        goto LABEL_18;
    }
    if ( v19 == v18 )
      goto LABEL_18;
    v22 = 0;
    do
    {
      v22 += sub_11FCB90(v20, v9);
      v29 = v21 + 1;
      if ( v21 + 1 == v19 )
        break;
      while ( 1 )
      {
        v20 = *v29;
        v21 = v29;
        if ( (unsigned __int64)*v29 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v19 == ++v29 )
          goto LABEL_19;
      }
    }
    while ( v19 != v29 );
LABEL_19:
    if ( *(_QWORD *)(a1 + 48) <= v22 )
      *(_QWORD *)(a1 + 48) = v22;
    v23 = *(_DWORD *)(a1 + 40);
    if ( !v23 )
    {
      ++*(_QWORD *)(a1 + 16);
LABEL_58:
      sub_A2B080(v61, 2 * v23);
      v38 = *(_DWORD *)(a1 + 40);
      if ( v38 )
      {
        a5 = (unsigned int)(v38 - 1);
        a6 = *(_QWORD *)(a1 + 24);
        v39 = a5 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v32 = *(_DWORD *)(a1 + 32) + 1;
        v27 = (__int64 *)(a6 + 16LL * v39);
        v40 = *v27;
        if ( v9 != *v27 )
        {
          v41 = 1;
          v42 = 0;
          while ( v40 != -4096 )
          {
            if ( v40 == -8192 && !v42 )
              v42 = v27;
            v39 = a5 & (v41 + v39);
            v27 = (__int64 *)(a6 + 16LL * v39);
            v40 = *v27;
            if ( v9 == *v27 )
              goto LABEL_48;
            ++v41;
          }
          if ( v42 )
            v27 = v42;
        }
        goto LABEL_48;
      }
LABEL_117:
      ++*(_DWORD *)(a1 + 32);
      BUG();
    }
    v24 = v23 - 1;
    a6 = *(_QWORD *)(a1 + 24);
    a5 = 1;
    v25 = 0;
    v26 = (v23 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v27 = (__int64 *)(a6 + 16LL * v26);
    v28 = *v27;
    if ( v9 == *v27 )
      goto LABEL_23;
    while ( v28 != -4096 )
    {
      if ( v28 == -8192 && !v25 )
        v25 = v27;
      v30 = a5 + 1;
      a5 = v26 + (unsigned int)a5;
      v26 = v24 & a5;
      v27 = (__int64 *)(a6 + 16LL * (v24 & (unsigned int)a5));
      v28 = *v27;
      if ( v9 == *v27 )
        goto LABEL_23;
      a5 = v30;
    }
    v31 = *(_DWORD *)(a1 + 32);
    if ( v25 )
      v27 = v25;
    ++*(_QWORD *)(a1 + 16);
    v32 = v31 + 1;
    if ( 4 * (v31 + 1) >= 3 * v23 )
      goto LABEL_58;
    if ( v23 - *(_DWORD *)(a1 + 36) - v32 <= v23 >> 3 )
    {
      sub_A2B080(v61, v23);
      v33 = *(_DWORD *)(a1 + 40);
      if ( v33 )
      {
        a5 = (unsigned int)(v33 - 1);
        a6 = *(_QWORD *)(a1 + 24);
        v34 = 1;
        v35 = a5 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v32 = *(_DWORD *)(a1 + 32) + 1;
        v36 = 0;
        v27 = (__int64 *)(a6 + 16LL * v35);
        v37 = *v27;
        if ( v9 != *v27 )
        {
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v36 )
              v36 = v27;
            v35 = a5 & (v34 + v35);
            v27 = (__int64 *)(a6 + 16LL * v35);
            v37 = *v27;
            if ( v9 == *v27 )
              goto LABEL_48;
            ++v34;
          }
          if ( v36 )
            v27 = v36;
        }
        goto LABEL_48;
      }
      goto LABEL_117;
    }
LABEL_48:
    *(_DWORD *)(a1 + 32) = v32;
    if ( *v27 != -4096 )
      --*(_DWORD *)(a1 + 36);
    *v27 = v9;
    v27[1] = 0;
LABEL_23:
    v27[1] = v22;
    if ( !v69 )
      _libc_free((unsigned __int64)v66);
    result = *(_QWORD *)(v63 + 8);
    v63 = result;
  }
  while ( v62 != result );
LABEL_26:
  if ( !(_BYTE)qword_4FDB148 )
  {
    v43 = *(_QWORD *)(a1 + 8);
    result = v43 + 16;
    if ( *(_QWORD *)(v43 + 32) != v43 + 16 )
    {
      v64 = *(_QWORD *)(v43 + 32);
      do
      {
        v44 = *(_QWORD *)(v64 + 40);
        while ( 1 )
        {
          v65 = 0;
          v66 = (__int64 *)v70;
          v67 = 16;
          v68 = 0;
          v69 = 1;
          v45 = *(_QWORD *)(v44 + 16);
          v46 = *(_QWORD *)(v44 + 24);
          if ( v45 == v46 )
            break;
          v47 = *(_QWORD *)(v44 + 16);
          v48 = 1;
          while ( 1 )
          {
            v49 = *(__int64 **)(v47 + 32);
            v50 = v49[1];
            if ( !v48 )
              goto LABEL_83;
            v51 = v66;
            v49 = &v66[HIDWORD(v67)];
            if ( v66 != v49 )
              break;
LABEL_89:
            if ( HIDWORD(v67) < (unsigned int)v67 )
            {
              ++HIDWORD(v67);
              *v49 = v50;
              v56 = (unsigned __int64)v66;
              ++v65;
              v48 = v69;
              goto LABEL_84;
            }
LABEL_83:
            sub_C8CC70((__int64)&v65, v50, (__int64)v49, v45, a5, a6);
            v48 = v69;
            v56 = (unsigned __int64)v66;
            if ( !v57 )
              goto LABEL_75;
LABEL_84:
            v47 += 40;
            if ( v46 == v47 )
            {
              if ( !v48 )
                _libc_free(v56);
              goto LABEL_87;
            }
          }
          while ( v50 != *v51 )
          {
            if ( v49 == ++v51 )
              goto LABEL_89;
          }
LABEL_75:
          --*(_DWORD *)(*(_QWORD *)(v47 + 32) + 40LL);
          v52 = *(_QWORD *)(v44 + 24);
          if ( *(_BYTE *)(v47 + 24) )
          {
            v53 = *(_QWORD *)(v47 + 16);
            if ( *(_BYTE *)(v52 - 16) )
            {
              v60 = *(_QWORD *)(v52 - 24);
              if ( v53 != v60 )
              {
                if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
                {
                  sub_BD60C0((_QWORD *)v47);
                  v60 = *(_QWORD *)(v52 - 24);
                }
                *(_QWORD *)(v47 + 16) = v60;
                if ( v60 != 0 && v60 != -4096 && v60 != -8192 )
                  sub_BD6050((unsigned __int64 *)v47, *(_QWORD *)(v52 - 40) & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
            else
            {
              *(_BYTE *)(v47 + 24) = 0;
              if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
                sub_BD60C0((_QWORD *)v47);
            }
          }
          else if ( *(_BYTE *)(v52 - 16) )
          {
            *(_QWORD *)v47 = 6;
            v58 = *(_QWORD *)(v52 - 24);
            *(_QWORD *)(v47 + 8) = 0;
            *(_QWORD *)(v47 + 16) = v58;
            if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
              sub_BD6050((unsigned __int64 *)v47, *(_QWORD *)(v52 - 40) & 0xFFFFFFFFFFFFFFF8LL);
            *(_BYTE *)(v47 + 24) = 1;
          }
          *(_QWORD *)(v47 + 32) = *(_QWORD *)(v52 - 8);
          v54 = *(_QWORD *)(v44 + 24);
          v55 = (_QWORD *)(v54 - 40);
          *(_QWORD *)(v44 + 24) = v54 - 40;
          if ( *(_BYTE *)(v54 - 16) )
          {
            *(_BYTE *)(v54 - 16) = 0;
            v59 = *(_QWORD *)(v54 - 24);
            if ( v59 != 0 && v59 != -4096 && v59 != -8192 )
              sub_BD60C0(v55);
          }
          if ( !v69 )
            _libc_free((unsigned __int64)v66);
        }
LABEL_87:
        result = sub_220EEE0(v64);
        v64 = result;
      }
      while ( v43 + 16 != result );
    }
  }
  return result;
}
