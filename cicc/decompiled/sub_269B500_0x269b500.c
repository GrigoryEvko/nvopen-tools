// Function: sub_269B500
// Address: 0x269b500
//
__int64 __fastcall sub_269B500(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // r9
  __int64 *v6; // r12
  __int64 *v7; // r15
  __int64 v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  int i; // ebx
  unsigned int v21; // r12d
  char *v22; // r14
  __int64 v23; // r13
  __int64 j; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int8 *v27; // r9
  unsigned __int8 *v28; // r14
  int v29; // r10d
  __int64 *v30; // r8
  _QWORD *v31; // rsi
  unsigned int v32; // eax
  unsigned __int8 *v33; // r11
  int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // [rsp+10h] [rbp-1A0h]
  __int64 v39; // [rsp+18h] [rbp-198h]
  __int64 v40; // [rsp+30h] [rbp-180h]
  __int64 v41; // [rsp+30h] [rbp-180h]
  __int64 *v42; // [rsp+38h] [rbp-178h]
  __int64 *v43; // [rsp+38h] [rbp-178h]
  _QWORD v44[2]; // [rsp+40h] [rbp-170h] BYREF
  _QWORD v45[2]; // [rsp+50h] [rbp-160h] BYREF
  _QWORD v46[2]; // [rsp+60h] [rbp-150h] BYREF
  char v47; // [rsp+70h] [rbp-140h] BYREF
  _QWORD v48[7]; // [rsp+90h] [rbp-120h] BYREF
  int v49; // [rsp+C8h] [rbp-E8h]
  char v50; // [rsp+CCh] [rbp-E4h] BYREF
  __int64 v51; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v52; // [rsp+D8h] [rbp-D8h]
  __int64 v53; // [rsp+E0h] [rbp-D0h]
  __int64 v54; // [rsp+E8h] [rbp-C8h]
  _QWORD *v55; // [rsp+F0h] [rbp-C0h]
  __int64 v56; // [rsp+F8h] [rbp-B8h]
  _BYTE v57[176]; // [rsp+100h] [rbp-B0h] BYREF

  v1 = a1;
  v48[0] = 0x1400000012LL;
  v48[1] = 0x1A00000016LL;
  v48[2] = 0x1D0000001CLL;
  v48[3] = 0x1F0000001ELL;
  v48[4] = 0x2100000020LL;
  v48[5] = 0x2300000022LL;
  v48[6] = 0x2600000025LL;
  v45[1] = &v51;
  v44[0] = &v51;
  v2 = *(_QWORD *)(a1 + 72);
  v55 = v57;
  v38 = v2;
  v40 = v2 + 4312;
  v3 = *(_QWORD *)(a1 + 40);
  v56 = 0x1000000000LL;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v44[1] = a1;
  v45[0] = v44;
  v4 = *(unsigned int *)(v3 + 8);
  v51 = 0;
  v5 = *(__int64 **)v3;
  v49 = 39;
  v6 = &v5[v4];
  if ( v5 != v6 )
  {
    v7 = v5;
    do
    {
      v8 = *v7;
      v46[0] = &v47;
      v46[1] = 0x800000000LL;
      v9 = sub_267FA80(v40, v8);
      v10 = *v9 + 8LL * *((unsigned int *)v9 + 2);
      if ( *v9 != v10 )
      {
        v42 = v6;
        v11 = *v9;
        do
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
            if ( *(_BYTE *)v12 == 85 && *(_QWORD *)v11 == v12 - 32 )
            {
              if ( *(char *)(v12 + 7) >= 0 )
                goto LABEL_52;
              v39 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
              v13 = sub_BD2BC0(v39);
              v12 = v39;
              v15 = v13 + v14;
              v16 = 0;
              if ( *(char *)(v39 + 7) < 0 )
              {
                v16 = sub_BD2BC0(v39);
                v12 = v39;
              }
              if ( !(unsigned int)((v15 - v16) >> 4) )
              {
LABEL_52:
                v17 = *(_QWORD *)(v38 + 4432);
                if ( v17 )
                {
                  v18 = *(_QWORD *)(v12 - 32);
                  if ( v18 )
                  {
                    if ( !*(_BYTE *)v18 && *(_QWORD *)(v18 + 24) == *(_QWORD *)(v12 + 80) && v17 == v18 )
                      break;
                  }
                }
              }
            }
            v11 += 8;
            if ( v10 == v11 )
              goto LABEL_18;
          }
          v11 += 8;
          sub_269AE30((__int64)v45, *(_QWORD *)(v12 + 16));
        }
        while ( v10 != v11 );
LABEL_18:
        v6 = v42;
      }
      ++v7;
    }
    while ( v6 != v7 );
    v1 = a1;
    v19 = 0;
    for ( i = 0; (unsigned int)v56 > (unsigned int)v19; i = v19 )
    {
      sub_269AE30((__int64)v45, *(_QWORD *)(v55[v19] + 16LL));
      v19 = (unsigned int)(i + 1);
    }
    v3 = *(_QWORD *)(a1 + 40);
  }
  v41 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  if ( *(_QWORD *)v3 == v41 )
  {
    v21 = 0;
    goto LABEL_38;
  }
  v43 = *(__int64 **)v3;
  v21 = 0;
  do
  {
    v22 = (char *)v48;
    v23 = *v43;
    for ( j = 18; ; j = *(int *)v22 )
    {
      v22 += 4;
      v21 |= sub_2683090(v1, v23, *(_QWORD *)(v1 + 72) + 160 * j + 3512, 0);
      if ( v22 == &v50 )
        break;
    }
    if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v23, v23, v25, v26);
      v27 = *(unsigned __int8 **)(v23 + 96);
      v28 = &v27[40 * *(_QWORD *)(v23 + 104)];
      if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v23, v23, v36, v37);
        v27 = *(unsigned __int8 **)(v23 + 96);
      }
    }
    else
    {
      v27 = *(unsigned __int8 **)(v23 + 96);
      v28 = &v27[40 * *(_QWORD *)(v23 + 104)];
    }
    v29 = v53;
    v30 = v46;
    if ( v27 == v28 )
    {
LABEL_41:
      v33 = 0;
      goto LABEL_37;
    }
    while ( 1 )
    {
      v46[0] = v27;
      if ( !v29 )
        break;
      if ( (_DWORD)v54 )
      {
        v32 = (v54 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v33 = *(unsigned __int8 **)(v52 + 8LL * v32);
        if ( v27 == v33 )
          goto LABEL_37;
        v35 = 1;
        while ( v33 != (unsigned __int8 *)-4096LL )
        {
          v32 = (v54 - 1) & (v35 + v32);
          v33 = *(unsigned __int8 **)(v52 + 8LL * v32);
          if ( v27 == v33 )
            goto LABEL_37;
          ++v35;
        }
      }
LABEL_33:
      v27 += 40;
      if ( v28 == v27 )
        goto LABEL_41;
    }
    v31 = &v55[(unsigned int)v56];
    if ( v31 == sub_266E410(v55, (__int64)v31, v30) )
      goto LABEL_33;
    v33 = v27;
LABEL_37:
    ++v43;
    v21 |= sub_2683090(v1, v23, *(_QWORD *)(v1 + 72) + 4312LL, v33);
  }
  while ( (__int64 *)v41 != v43 );
LABEL_38:
  if ( v55 != (_QWORD *)v57 )
    _libc_free((unsigned __int64)v55);
  sub_C7D6A0(v52, 8LL * (unsigned int)v54, 8);
  return v21;
}
