// Function: sub_23CB250
// Address: 0x23cb250
//
void __fastcall sub_23CB250(_QWORD *a1)
{
  __int64 v1; // rdx
  _QWORD *v3; // rdi
  unsigned int v4; // ebx
  __int64 v5; // rax
  unsigned int v6; // ecx
  __int64 *v7; // rsi
  __int64 v8; // r11
  __int64 v9; // r13
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // rsi
  _BYTE *v15; // rsi
  int v16; // esi
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // edx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 v24; // r9
  __int64 v25; // r15
  __int64 v26; // r8
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // r11d
  unsigned int v31; // ecx
  _QWORD *v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  int v36; // r13d
  int v37; // edx
  _QWORD *v38; // rdi
  unsigned int v39; // r14d
  int v40; // ecx
  __int64 v41; // rsi
  unsigned int v42; // ecx
  __int64 v43; // r14
  int v44; // edi
  _QWORD *v45; // rsi
  __int64 v46; // [rsp+0h] [rbp-D0h]
  __int64 v47; // [rsp+0h] [rbp-D0h]
  unsigned int v48; // [rsp+8h] [rbp-C8h]
  __int64 v49; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+8h] [rbp-C8h]
  __int64 v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+18h] [rbp-B8h]
  __int64 v53; // [rsp+38h] [rbp-98h] BYREF
  __int64 v54; // [rsp+40h] [rbp-90h] BYREF
  __int64 v55; // [rsp+48h] [rbp-88h]
  __int64 v56; // [rsp+50h] [rbp-80h]
  unsigned int v57; // [rsp+58h] [rbp-78h]
  _QWORD *v58; // [rsp+60h] [rbp-70h] BYREF
  __int64 v59; // [rsp+68h] [rbp-68h]
  _QWORD v60[12]; // [rsp+70h] [rbp-60h] BYREF

  LODWORD(v1) = 1;
  v3 = v60;
  v4 = 0;
  v5 = a1[27];
  v58 = v60;
  v60[0] = v5;
  v59 = 0x600000001LL;
  v52 = (__int64)(a1 + 31);
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  while ( 1 )
  {
    v12 = (unsigned int)v1;
    v1 = (unsigned int)(v1 - 1);
    v13 = v3[v12 - 1];
    LODWORD(v59) = v1;
    if ( *(_DWORD *)(v13 + 8) != 1 )
      break;
    if ( v57 )
    {
      v6 = (v57 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v7 = (__int64 *)(v55 + 24LL * v6);
      v8 = *v7;
      if ( v13 == *v7 )
      {
LABEL_4:
        if ( v7 != (__int64 *)(v55 + 24LL * v57) )
        {
          v9 = v7[2];
          v10 = sub_23CC790(v7[1]);
          sub_23CC7B0(v13, v10);
          v11 = sub_23CC780(v9);
          sub_23CC7A0(v13, v11);
          LODWORD(v1) = v59;
          goto LABEL_6;
        }
      }
      else
      {
        v16 = 1;
        while ( v8 != -4096 )
        {
          v36 = v16 + 1;
          v6 = (v57 - 1) & (v16 + v6);
          v7 = (__int64 *)(v55 + 24LL * v6);
          v8 = *v7;
          if ( v13 == *v7 )
            goto LABEL_4;
          v16 = v36;
        }
      }
    }
    if ( !*(_DWORD *)(v13 + 56) )
      goto LABEL_6;
    v17 = *(_QWORD *)(v13 + 48);
    v18 = v17 + 16LL * *(unsigned int *)(v13 + 64);
    if ( v17 == v18 )
      goto LABEL_6;
    while ( 1 )
    {
      v19 = v17;
      if ( *(_DWORD *)v17 <= 0xFFFFFFFD )
        break;
      v17 += 16;
      if ( v18 == v17 )
        goto LABEL_6;
    }
    if ( v17 == v18 )
      goto LABEL_6;
    if ( v12 > HIDWORD(v59) )
    {
      v47 = v17;
      v49 = v18;
      sub_C8D5F0((__int64)&v58, v60, v12, 8u, v12, v17);
      v3 = v58;
      v1 = (unsigned int)v59;
      v17 = v47;
      v18 = v49;
    }
    v3[v1] = v13;
    v20 = v59 + 1;
    v21 = *(unsigned int *)(v13 + 64);
    LODWORD(v59) = v59 + 1;
    v22 = *(_QWORD *)(v17 + 8);
    v23 = v22;
    if ( v17 == *(_QWORD *)(v13 + 48) + 16 * v21 )
    {
      v23 = 0;
    }
    else
    {
      v24 = (__int64)a1;
      v25 = v22;
      v26 = v4;
      v27 = v18;
      v28 = v20;
      v29 = v20 + 1LL;
      if ( v29 > HIDWORD(v59) )
        goto LABEL_34;
      while ( 1 )
      {
        v19 += 16;
        v58[v28] = v23;
        v28 = (unsigned int)(v59 + 1);
        for ( LODWORD(v59) = v59 + 1; v27 != v19; v19 += 16 )
        {
          if ( *(_DWORD *)v19 <= 0xFFFFFFFD )
            break;
        }
        if ( v19 == *(_QWORD *)(v13 + 48) + 16LL * *(unsigned int *)(v13 + 64) )
          break;
        v29 = v28 + 1;
        v23 = *(_QWORD *)(v19 + 8);
        if ( v28 + 1 > (unsigned __int64)HIDWORD(v59) )
        {
LABEL_34:
          v46 = v24;
          v48 = v26;
          sub_C8D5F0((__int64)&v58, v60, v29, 8u, v26, v24);
          v28 = (unsigned int)v59;
          v24 = v46;
          v26 = v48;
        }
      }
      v4 = v26;
      v22 = v25;
      a1 = (_QWORD *)v24;
    }
    if ( !v57 )
    {
      ++v54;
      goto LABEL_64;
    }
    v30 = 1;
    v31 = (v57 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v32 = (_QWORD *)(v55 + 24LL * v31);
    v33 = 0;
    v34 = *v32;
    if ( v13 != *v32 )
    {
      while ( v34 != -4096 )
      {
        if ( !v33 && v34 == -8192 )
          v33 = v32;
        v31 = (v57 - 1) & (v30 + v31);
        v32 = (_QWORD *)(v55 + 24LL * v31);
        v34 = *v32;
        if ( v13 == *v32 )
          goto LABEL_38;
        ++v30;
      }
      if ( !v33 )
        v33 = v32;
      ++v54;
      v37 = v56 + 1;
      if ( 4 * ((int)v56 + 1) >= 3 * v57 )
      {
LABEL_64:
        v51 = v22;
        sub_23CB050((__int64)&v54, 2 * v57);
        if ( !v57 )
        {
LABEL_81:
          LODWORD(v56) = v56 + 1;
          BUG();
        }
        v22 = v51;
        v37 = v56 + 1;
        v42 = (v57 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v33 = (_QWORD *)(v55 + 24LL * v42);
        v43 = *v33;
        if ( v13 != *v33 )
        {
          v44 = 1;
          v45 = 0;
          while ( v43 != -4096 )
          {
            if ( !v45 && v43 == -8192 )
              v45 = v33;
            v42 = (v57 - 1) & (v44 + v42);
            v33 = (_QWORD *)(v55 + 24LL * v42);
            v43 = *v33;
            if ( v13 == *v33 )
              goto LABEL_54;
            ++v44;
          }
          if ( v45 )
            v33 = v45;
        }
      }
      else if ( v57 - HIDWORD(v56) - v37 <= v57 >> 3 )
      {
        v50 = v22;
        sub_23CB050((__int64)&v54, v57);
        if ( !v57 )
          goto LABEL_81;
        v38 = 0;
        v39 = (v57 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v22 = v50;
        v37 = v56 + 1;
        v40 = 1;
        v33 = (_QWORD *)(v55 + 24LL * v39);
        v41 = *v33;
        if ( v13 != *v33 )
        {
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v38 )
              v38 = v33;
            v39 = (v57 - 1) & (v40 + v39);
            v33 = (_QWORD *)(v55 + 24LL * v39);
            v41 = *v33;
            if ( v13 == *v33 )
              goto LABEL_54;
            ++v40;
          }
          if ( v38 )
            v33 = v38;
        }
      }
LABEL_54:
      LODWORD(v56) = v37;
      if ( *v33 != -4096 )
        --HIDWORD(v56);
      *v33 = v13;
      v35 = v33 + 1;
      *v35 = 0;
      v35[1] = 0;
      goto LABEL_39;
    }
LABEL_38:
    v35 = v32 + 1;
LABEL_39:
    *v35 = v22;
    v35[1] = v23;
    LODWORD(v1) = v59;
LABEL_6:
    if ( !(_DWORD)v1 )
      goto LABEL_14;
LABEL_7:
    v3 = v58;
  }
  sub_23CC7A0(v13, v4);
  v14 = v4++;
  sub_23CC7B0(v13, v14);
  v53 = v13;
  v15 = (_BYTE *)a1[32];
  if ( v15 == (_BYTE *)a1[33] )
  {
    sub_23CACE0(v52, v15, &v53);
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v13;
      v15 = (_BYTE *)a1[32];
    }
    a1[32] = v15 + 8;
  }
  LODWORD(v1) = v59;
  if ( (_DWORD)v59 )
    goto LABEL_7;
LABEL_14:
  sub_C7D6A0(v55, 24LL * v57, 8);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
}
