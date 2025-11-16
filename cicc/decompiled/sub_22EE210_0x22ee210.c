// Function: sub_22EE210
// Address: 0x22ee210
//
__int64 __fastcall sub_22EE210(__int64 a1)
{
  __int64 v1; // rsi
  char v2; // r13
  __int64 v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // r12
  int v8; // r15d
  __int64 *v9; // r11
  unsigned int v10; // r10d
  __int64 v11; // r9
  __int64 v12; // rcx
  unsigned int v13; // r12d
  int v15; // eax
  unsigned __int8 v16; // al
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // r14
  _QWORD *v21; // r15
  __int64 v22; // rdx
  __int64 v23; // r14
  unsigned __int64 v24; // r8
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 *v31; // rdi
  unsigned int v32; // r14d
  __int64 v33; // rcx
  bool v34; // zf
  __int64 v35; // [rsp+10h] [rbp-160h] BYREF
  __int64 v36; // [rsp+18h] [rbp-158h]
  __int64 v37; // [rsp+20h] [rbp-150h]
  __int64 v38; // [rsp+28h] [rbp-148h]
  _QWORD *v39; // [rsp+30h] [rbp-140h] BYREF
  __int64 v40; // [rsp+38h] [rbp-138h]
  _QWORD v41[38]; // [rsp+40h] [rbp-130h] BYREF

  v1 = 0;
  v2 = 1;
  v39 = v41;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v41[0] = a1;
  v3 = 0;
  v40 = 0x2000000001LL;
  v4 = 1;
  while ( 1 )
  {
    v5 = (unsigned int)(v1 - 1);
    v6 = 8LL * v4 - 8;
    if ( !v4 )
    {
LABEL_6:
      v13 = (v2 == 0) + 2;
      goto LABEL_7;
    }
    while ( 1 )
    {
      --v4;
      v7 = *(_QWORD *)((char *)v39 + v6);
      LODWORD(v40) = v4;
      if ( !(_DWORD)v1 )
      {
        ++v35;
        goto LABEL_58;
      }
      v8 = 1;
      v9 = 0;
      v10 = v5 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = v3 + 8LL * v10;
      v12 = *(_QWORD *)v11;
      if ( v7 != *(_QWORD *)v11 )
        break;
LABEL_5:
      v6 -= 8;
      if ( !v4 )
        goto LABEL_6;
    }
    while ( v12 != -4096 )
    {
      if ( v12 != -8192 || v9 )
        v11 = (__int64)v9;
      v10 = v5 & (v8 + v10);
      v12 = *(_QWORD *)(v3 + 8LL * v10);
      if ( v7 == v12 )
        goto LABEL_5;
      ++v8;
      v9 = (__int64 *)v11;
      v11 = v3 + 8LL * v10;
    }
    if ( !v9 )
      v9 = (__int64 *)v11;
    ++v35;
    v15 = v37 + 1;
    if ( 4 * ((int)v37 + 1) < (unsigned int)(3 * v1) )
    {
      if ( (int)v1 - (v15 + HIDWORD(v37)) > (unsigned int)v1 >> 3 )
        goto LABEL_19;
      sub_BD14B0((__int64)&v35, v1);
      if ( (_DWORD)v38 )
      {
        v1 = v36;
        v5 = 1;
        v31 = 0;
        v32 = (v38 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v9 = (__int64 *)(v36 + 8LL * v32);
        v33 = *v9;
        v15 = v37 + 1;
        if ( v7 != *v9 )
        {
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v31 )
              v31 = v9;
            v11 = (unsigned int)(v5 + 1);
            v32 = (v38 - 1) & (v5 + v32);
            v9 = (__int64 *)(v36 + 8LL * v32);
            v33 = *v9;
            if ( v7 == *v9 )
              goto LABEL_19;
            v5 = (unsigned int)v11;
          }
          if ( v31 )
            v9 = v31;
        }
        goto LABEL_19;
      }
LABEL_84:
      LODWORD(v37) = v37 + 1;
      BUG();
    }
LABEL_58:
    sub_BD14B0((__int64)&v35, 2 * v1);
    if ( !(_DWORD)v38 )
      goto LABEL_84;
    v30 = (v38 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v9 = (__int64 *)(v36 + 8LL * v30);
    v1 = *v9;
    v15 = v37 + 1;
    if ( v7 != *v9 )
    {
      v11 = 1;
      v5 = 0;
      while ( v1 != -4096 )
      {
        if ( v1 == -8192 && !v5 )
          v5 = (__int64)v9;
        v30 = (v38 - 1) & (v11 + v30);
        v9 = (__int64 *)(v36 + 8LL * v30);
        v1 = *v9;
        if ( v7 == *v9 )
          goto LABEL_19;
        v11 = (unsigned int)(v11 + 1);
      }
      if ( v5 )
        v9 = (__int64 *)v5;
    }
LABEL_19:
    LODWORD(v37) = v15;
    if ( *v9 != -4096 )
      --HIDWORD(v37);
    *v9 = v7;
    v16 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 > 0x1Cu )
      break;
    if ( v16 > 0x15u )
      goto LABEL_23;
    v34 = v7 == sub_AD6530(*(_QWORD *)(v7 + 8), v1);
    v4 = v40;
    if ( !v34 )
      v2 = 0;
LABEL_28:
    v3 = v36;
    v1 = (unsigned int)v38;
  }
  if ( (unsigned int)v16 - 67 <= 0xC )
  {
    v17 = (__int64)sub_BD3990((unsigned __int8 *)v7, v1);
    goto LABEL_26;
  }
  switch ( v16 )
  {
    case '?':
      v17 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
      goto LABEL_26;
    case 'T':
      v20 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      {
        v21 = *(_QWORD **)(v7 - 8);
        v7 = (__int64)&v21[(unsigned __int64)v20 / 8];
      }
      else
      {
        v21 = (_QWORD *)(v7 - v20);
      }
      v22 = (unsigned int)v40;
      v23 = v20 >> 5;
      v24 = (unsigned int)v40 + v23;
      if ( v24 > HIDWORD(v40) )
      {
        sub_C8D5F0((__int64)&v39, v41, (unsigned int)v40 + v23, 8u, v24, v11);
        v22 = (unsigned int)v40;
      }
      v25 = &v39[v22];
      if ( v21 != (_QWORD *)v7 )
      {
        do
        {
          if ( v25 )
            *v25 = *v21;
          v21 += 4;
          ++v25;
        }
        while ( v21 != (_QWORD *)v7 );
        LODWORD(v22) = v40;
      }
      LODWORD(v40) = v22 + v23;
      v4 = v22 + v23;
      goto LABEL_28;
    case 'V':
      v26 = (unsigned int)v40;
      v27 = *(_QWORD *)(v7 - 64);
      v28 = (unsigned int)v40 + 1LL;
      if ( v28 > HIDWORD(v40) )
      {
        sub_C8D5F0((__int64)&v39, v41, v28, 8u, v5, v11);
        v26 = (unsigned int)v40;
      }
      v39[v26] = v27;
      v18 = (unsigned int)(v40 + 1);
      v19 = v18 + 1;
      LODWORD(v40) = v40 + 1;
      v17 = *(_QWORD *)(v7 - 32);
      if ( v18 + 1 <= (unsigned __int64)HIDWORD(v40) )
        goto LABEL_27;
      goto LABEL_49;
  }
  if ( v16 != 85 )
  {
    if ( v16 != 96 )
      goto LABEL_23;
    v17 = *(_QWORD *)(v7 - 32);
LABEL_26:
    v18 = (unsigned int)v40;
    v19 = (unsigned int)v40 + 1LL;
    if ( v19 <= HIDWORD(v40) )
    {
LABEL_27:
      v39[v18] = v17;
      v4 = v40 + 1;
      LODWORD(v40) = v40 + 1;
      goto LABEL_28;
    }
LABEL_49:
    sub_C8D5F0((__int64)&v39, v41, v19, 8u, v5, v11);
    v18 = (unsigned int)v40;
    goto LABEL_27;
  }
  v29 = *(_QWORD *)(v7 - 32);
  if ( v29
    && !*(_BYTE *)v29
    && *(_QWORD *)(v29 + 24) == *(_QWORD *)(v7 + 80)
    && (*(_BYTE *)(v29 + 33) & 0x20) != 0
    && *(_DWORD *)(v29 + 36) == 149 )
  {
    v17 = sub_B5B890(v7);
    goto LABEL_26;
  }
LABEL_23:
  v1 = (unsigned int)v38;
  v3 = v36;
  v13 = 1;
LABEL_7:
  sub_C7D6A0(v3, 8 * v1, 8);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  return v13;
}
