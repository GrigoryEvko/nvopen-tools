// Function: sub_2787550
// Address: 0x2787550
//
__int64 __fastcall sub_2787550(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rsi
  __int64 v4; // r14
  unsigned __int64 v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // eax
  unsigned int v18; // r13d
  __int64 *v19; // rcx
  __int64 v20; // rdi
  __int64 v21; // r9
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  int v24; // r8d
  unsigned int v25; // edx
  _QWORD *v26; // rax
  _BYTE *v27; // r10
  _BYTE *v28; // rcx
  int v30; // eax
  int v31; // r11d
  int v32; // edx
  int v33; // r8d
  int v34; // ecx
  int v35; // r8d
  __int64 v36; // r12
  unsigned int v37; // ebx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r14
  unsigned __int64 v46; // rbx
  unsigned __int8 v48; // [rsp+Fh] [rbp-B1h]
  __int64 v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+20h] [rbp-A0h]
  __int64 v51; // [rsp+28h] [rbp-98h]
  __int64 v52; // [rsp+48h] [rbp-78h] BYREF
  unsigned __int64 v53; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-68h]
  unsigned __int64 v55; // [rsp+60h] [rbp-60h]
  unsigned int v56; // [rsp+68h] [rbp-58h]
  unsigned __int64 v57; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v58; // [rsp+78h] [rbp-48h]
  unsigned __int64 v59; // [rsp+80h] [rbp-40h]
  unsigned int v60; // [rsp+88h] [rbp-38h]

  v49 = a1 + 168;
  v51 = *(_QWORD *)(a1 + 184);
  if ( v51 == a1 + 168 )
    return 0;
  v48 = 0;
  v2 = a1;
  do
  {
    v3 = (unsigned int)(qword_4FFB348 + 1);
    sub_AADB10((__int64)&v53, v3, 0);
    if ( (*(_BYTE *)(v51 + 40) & 1) == 0 )
    {
LABEL_34:
      if ( v56 > 0x40 && v55 )
        j_j___libc_free_0_0(v55);
      if ( v54 > 0x40 && v53 )
        j_j___libc_free_0_0(v53);
      goto LABEL_40;
    }
    v4 = v2;
    v50 = 0;
    v5 = v51 + 32;
    do
    {
      v7 = *(_QWORD *)(v5 + 16);
      v8 = *(unsigned int *)(v4 + 24);
      v9 = *(_QWORD *)(v4 + 8);
      v52 = v7;
      if ( !(_DWORD)v8 )
        goto LABEL_11;
      v3 = ((_DWORD)v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v10 = (__int64 *)(v9 + 16 * v3);
      v11 = *v10;
      if ( v7 != *v10 )
      {
        v32 = 1;
        while ( v11 != -4096 )
        {
          v33 = v32 + 1;
          v3 = ((_DWORD)v8 - 1) & (unsigned int)(v32 + v3);
          v10 = (__int64 *)(v9 + 16LL * (unsigned int)v3);
          v11 = *v10;
          if ( v7 == *v10 )
            goto LABEL_14;
          v32 = v33;
        }
        goto LABEL_11;
      }
LABEL_14:
      if ( v10 == (__int64 *)(v9 + 16 * v8) )
        goto LABEL_11;
      v12 = *(_QWORD *)(v4 + 32);
      v13 = v12 + 40LL * *((unsigned int *)v10 + 2);
      if ( v13 == v12 + 40LL * *(unsigned int *)(v4 + 40) )
        goto LABEL_11;
      sub_AB3510((__int64)&v57, (__int64)&v53, v13 + 8, 0);
      if ( v54 > 0x40 && v53 )
        j_j___libc_free_0_0(v53);
      v53 = v57;
      v14 = v58;
      v58 = 0;
      v54 = v14;
      if ( v56 > 0x40 && v55 )
      {
        j_j___libc_free_0_0(v55);
        v55 = v59;
        v56 = v60;
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        if ( *(_DWORD *)(v4 + 64) )
        {
LABEL_21:
          v15 = *(unsigned int *)(v4 + 72);
          v16 = *(_QWORD *)(v4 + 56);
          v3 = v16 + 8 * v15;
          if ( (_DWORD)v15 )
          {
            v17 = v15 - 1;
            v18 = (v15 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v19 = (__int64 *)(v16 + 8LL * v18);
            v20 = *v19;
            if ( v7 == *v19 )
            {
LABEL_23:
              if ( (__int64 *)v3 != v19 )
                goto LABEL_11;
            }
            else
            {
              v34 = 1;
              while ( v20 != -4096 )
              {
                v35 = v34 + 1;
                v18 = v17 & (v34 + v18);
                v19 = (__int64 *)(v16 + 8LL * v18);
                v20 = *v19;
                if ( v7 == *v19 )
                  goto LABEL_23;
                v34 = v35;
              }
            }
          }
          goto LABEL_24;
        }
      }
      else
      {
        v55 = v59;
        v56 = v60;
        if ( *(_DWORD *)(v4 + 64) )
          goto LABEL_21;
      }
      v6 = *(_QWORD **)(v4 + 80);
      v3 = (__int64)&v6[*(unsigned int *)(v4 + 88)];
      if ( (_QWORD *)v3 != sub_27849A0(v6, v3, &v52) )
        goto LABEL_11;
LABEL_24:
      if ( !v50 )
        v50 = *(_QWORD *)(v7 + 8);
      v21 = *(_QWORD *)(v7 + 16);
      if ( v21 )
      {
        while ( 1 )
        {
          v28 = *(_BYTE **)(v21 + 24);
          if ( *v28 <= 0x1Cu )
            break;
          v3 = *(_QWORD *)(v4 + 8);
          v22 = *(unsigned int *)(v4 + 24);
          v23 = (_QWORD *)(v3 + 16 * v22);
          if ( !(_DWORD)v22 )
            break;
          v24 = v22 - 1;
          v25 = (v22 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v26 = (_QWORD *)(v3 + 16LL * v25);
          v27 = (_BYTE *)*v26;
          if ( v28 != (_BYTE *)*v26 )
          {
            v30 = 1;
            while ( v27 != (_BYTE *)-4096LL )
            {
              v31 = v30 + 1;
              v25 = v24 & (v30 + v25);
              v26 = (_QWORD *)(v3 + 16LL * v25);
              v27 = (_BYTE *)*v26;
              if ( v28 == (_BYTE *)*v26 )
                goto LABEL_30;
              v30 = v31;
            }
            break;
          }
LABEL_30:
          if ( v23 == v26 )
            break;
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            goto LABEL_11;
        }
        v2 = v4;
        goto LABEL_34;
      }
LABEL_11:
      v5 = *(_QWORD *)(v5 + 8) & 0xFFFFFFFFFFFFFFFELL;
    }
    while ( v5 );
    v2 = v4;
    v36 = *(_QWORD *)(v51 + 40) & 1LL;
    if ( (*(_QWORD *)(v51 + 40) & 1) == 0 )
      goto LABEL_34;
    if ( sub_AAF760((__int64)&v53) )
      goto LABEL_34;
    if ( sub_AB0120((__int64)&v53) )
      goto LABEL_34;
    v37 = sub_AB1D50((__int64)&v53) + 1;
    v41 = sub_BCAC60(v50, v3, v38, v39, v40);
    if ( v37 > (unsigned int)sub_C336A0(v41) - 1 )
      goto LABEL_34;
    v45 = sub_AE44B0(a2, *(_QWORD *)(v4 + 256), v37);
    if ( !v45 )
    {
      if ( v37 <= 0x20 )
      {
        v45 = sub_BCB2D0(*(_QWORD **)(v2 + 256));
      }
      else
      {
        if ( v37 > 0x40 )
          goto LABEL_34;
        v45 = sub_BCB2E0(*(_QWORD **)(v2 + 256));
      }
    }
    if ( (*(_BYTE *)(v51 + 40) & 1) != 0 )
    {
      v46 = v51 + 32;
      do
      {
        sub_27867C0(v2, *(_QWORD *)(v46 + 16), v45, v42, v43, v44);
        v46 = *(_QWORD *)(v46 + 8) & 0xFFFFFFFFFFFFFFFELL;
      }
      while ( v46 );
    }
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
    if ( v54 > 0x40 && v53 )
      j_j___libc_free_0_0(v53);
    v48 = v36;
LABEL_40:
    v51 = sub_220EF30(v51);
  }
  while ( v49 != v51 );
  return v48;
}
