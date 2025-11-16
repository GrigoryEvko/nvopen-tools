// Function: sub_2809DC0
// Address: 0x2809dc0
//
void __fastcall sub_2809DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v7; // r11
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r9
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  _BYTE *v18; // rbx
  _BYTE *v19; // r12
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rcx
  int v26; // edx
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // edi
  unsigned int v30; // esi
  __int64 v31; // rax
  _QWORD *v32; // rcx
  int v33; // edi
  __int64 v34; // rdx
  unsigned __int64 *v35; // rdi
  __int64 v36; // rdx
  _QWORD *v37; // rdx
  __int64 v38; // r9
  __int64 v39; // r11
  _QWORD *v40; // rdx
  __int64 v41; // rdi
  int v42; // edi
  int v43; // edi
  _QWORD *v44; // rsi
  __int64 v45; // r9
  unsigned int v46; // edx
  __int64 v47; // r11
  int v48; // edi
  __int64 v49; // r9
  unsigned int v50; // edx
  __int64 v51; // r11
  _QWORD *v52; // [rsp+0h] [rbp-D0h]
  unsigned __int64 *v53; // [rsp+10h] [rbp-C0h]
  _QWORD *v54; // [rsp+10h] [rbp-C0h]
  _QWORD *v55; // [rsp+10h] [rbp-C0h]
  __int64 v56; // [rsp+10h] [rbp-C0h]
  int v57; // [rsp+10h] [rbp-C0h]
  __int64 v58; // [rsp+18h] [rbp-B8h]
  _QWORD v59[2]; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v60; // [rsp+38h] [rbp-98h]
  __int64 v61; // [rsp+40h] [rbp-90h]
  _BYTE *v62; // [rsp+50h] [rbp-80h] BYREF
  __int64 v63; // [rsp+58h] [rbp-78h]
  _BYTE v64[112]; // [rsp+60h] [rbp-70h] BYREF

  v62 = v64;
  v63 = 0x800000000LL;
  v5 = *(_QWORD *)(a1 + 120);
  v58 = *(_QWORD *)(v5 + 40);
  if ( v58 == *(_QWORD *)(v5 + 32) )
    return;
  v7 = *(_QWORD *)(v5 + 32);
  do
  {
    v8 = v7;
    v9 = *(_QWORD *)(*(_QWORD *)v7 + 56LL);
    v10 = *(_QWORD *)v7 + 48LL;
    if ( v10 == v9 )
      goto LABEL_17;
    do
    {
      v11 = v9 - 24;
      if ( !v9 )
        v11 = 0;
      if ( !*(_DWORD *)(a1 + 16) )
      {
        v12 = *(_QWORD **)(a1 + 32);
        v13 = 8LL * *(unsigned int *)(a1 + 40);
        v14 = &v12[(unsigned __int64)v13 / 8];
        v15 = v13 >> 3;
        v16 = v13 >> 5;
        if ( v16 )
        {
          v17 = &v12[4 * v16];
          while ( v11 != *v12 )
          {
            if ( v11 == v12[1] )
            {
              ++v12;
              break;
            }
            if ( v11 == v12[2] )
            {
              v12 += 2;
              break;
            }
            if ( v11 == v12[3] )
            {
              v12 += 3;
              break;
            }
            v12 += 4;
            if ( v17 == v12 )
            {
              v15 = v14 - v12;
              goto LABEL_27;
            }
          }
LABEL_14:
          if ( v14 != v12 )
            goto LABEL_15;
LABEL_30:
          if ( !*(_DWORD *)(a1 + 232) )
          {
            v22 = (unsigned int)v63;
            v23 = (unsigned int)v63 + 1LL;
            if ( v23 <= HIDWORD(v63) )
            {
LABEL_32:
              *(_QWORD *)&v62[8 * v22] = v11;
              LODWORD(v63) = v63 + 1;
              goto LABEL_15;
            }
LABEL_60:
            v56 = v11;
            sub_C8D5F0((__int64)&v62, v64, v23, 8u, a5, v11);
            v22 = (unsigned int)v63;
            v11 = v56;
            goto LABEL_32;
          }
          v59[1] = 0;
          a5 = a1 + 216;
          v59[0] = 2;
          v60 = v11;
          if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          {
            sub_BD73F0((__int64)v59);
            a5 = a1 + 216;
          }
          v30 = *(_DWORD *)(a1 + 240);
          v61 = a5;
          if ( v30 )
          {
            v31 = v60;
            v38 = *(_QWORD *)(a1 + 224);
            v39 = (v30 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v40 = (_QWORD *)(v38 + (v39 << 6));
            v41 = v40[3];
            if ( v41 == v60 )
            {
LABEL_71:
              v37 = v40 + 5;
              goto LABEL_56;
            }
            v57 = 1;
            v32 = 0;
            while ( v41 != -4096 )
            {
              if ( v41 == -8192 && !v32 )
                v32 = v40;
              LODWORD(v39) = (v30 - 1) & (v57 + v39);
              v40 = (_QWORD *)(v38 + ((unsigned __int64)(unsigned int)v39 << 6));
              v41 = v40[3];
              if ( v60 == v41 )
                goto LABEL_71;
              ++v57;
            }
            v42 = *(_DWORD *)(a1 + 232);
            if ( !v32 )
              v32 = v40;
            ++*(_QWORD *)(a1 + 216);
            v33 = v42 + 1;
            if ( 4 * v33 < 3 * v30 )
            {
              if ( v30 - *(_DWORD *)(a1 + 236) - v33 > v30 >> 3 )
              {
LABEL_47:
                *(_DWORD *)(a1 + 232) = v33;
                if ( v32[3] == -4096 )
                {
                  v35 = v32 + 1;
                  if ( v31 != -4096 )
                  {
LABEL_52:
                    v32[3] = v31;
                    if ( v31 == 0 || v31 == -4096 || v31 == -8192 )
                    {
                      v31 = v60;
                    }
                    else
                    {
                      v54 = v32;
                      sub_BD6050(v35, v59[0] & 0xFFFFFFFFFFFFFFF8LL);
                      v31 = v60;
                      v32 = v54;
                    }
                  }
                }
                else
                {
                  --*(_DWORD *)(a1 + 236);
                  v34 = v32[3];
                  if ( v34 != v31 )
                  {
                    v35 = v32 + 1;
                    LOBYTE(a5) = v34 != -4096;
                    if ( ((v34 != 0) & (unsigned __int8)a5) != 0 && v34 != -8192 )
                    {
                      v52 = v32;
                      v53 = v32 + 1;
                      sub_BD60C0(v35);
                      v31 = v60;
                      v32 = v52;
                      v35 = v53;
                    }
                    goto LABEL_52;
                  }
                }
                v36 = v61;
                v32[5] = 6;
                v32[6] = 0;
                v32[7] = 0;
                v32[4] = v36;
                v37 = v32 + 5;
LABEL_56:
                if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
                {
                  v55 = v37;
                  sub_BD60C0(v59);
                  v37 = v55;
                }
                v22 = (unsigned int)v63;
                v11 = v37[2];
                v23 = (unsigned int)v63 + 1LL;
                if ( v23 <= HIDWORD(v63) )
                  goto LABEL_32;
                goto LABEL_60;
              }
              sub_CF32C0(a5, v30);
              a5 = *(unsigned int *)(a1 + 240);
              if ( !(_DWORD)a5 )
                goto LABEL_45;
              v31 = v60;
              a5 = (unsigned int)(a5 - 1);
              v43 = 1;
              v44 = 0;
              v45 = *(_QWORD *)(a1 + 224);
              v46 = a5 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v32 = (_QWORD *)(v45 + ((unsigned __int64)v46 << 6));
              v47 = v32[3];
              if ( v60 == v47 )
                goto LABEL_46;
              while ( v47 != -4096 )
              {
                if ( !v44 && v47 == -8192 )
                  v44 = v32;
                v46 = a5 & (v43 + v46);
                v32 = (_QWORD *)(v45 + ((unsigned __int64)v46 << 6));
                v47 = v32[3];
                if ( v60 == v47 )
                  goto LABEL_46;
                ++v43;
              }
              goto LABEL_84;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 216);
          }
          sub_CF32C0(a5, 2 * v30);
          a5 = *(unsigned int *)(a1 + 240);
          if ( !(_DWORD)a5 )
          {
LABEL_45:
            v31 = v60;
            v32 = 0;
LABEL_46:
            v33 = *(_DWORD *)(a1 + 232) + 1;
            goto LABEL_47;
          }
          v31 = v60;
          a5 = (unsigned int)(a5 - 1);
          v48 = 1;
          v44 = 0;
          v49 = *(_QWORD *)(a1 + 224);
          v50 = a5 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v32 = (_QWORD *)(v49 + ((unsigned __int64)v50 << 6));
          v51 = v32[3];
          if ( v51 == v60 )
            goto LABEL_46;
          while ( v51 != -4096 )
          {
            if ( !v44 && v51 == -8192 )
              v44 = v32;
            v50 = a5 & (v48 + v50);
            v32 = (_QWORD *)(v49 + ((unsigned __int64)v50 << 6));
            v51 = v32[3];
            if ( v60 == v51 )
              goto LABEL_46;
            ++v48;
          }
LABEL_84:
          if ( v44 )
            v32 = v44;
          goto LABEL_46;
        }
LABEL_27:
        if ( v15 != 2 )
        {
          if ( v15 != 3 )
          {
            if ( v15 != 1 )
              goto LABEL_30;
            goto LABEL_65;
          }
          if ( v11 == *v12 )
            goto LABEL_14;
          ++v12;
        }
        if ( v11 == *v12 )
          goto LABEL_14;
        ++v12;
LABEL_65:
        if ( v11 != *v12 )
          goto LABEL_30;
        goto LABEL_14;
      }
      v24 = *(_DWORD *)(a1 + 24);
      v25 = *(_QWORD *)(a1 + 8);
      if ( !v24 )
        goto LABEL_30;
      v26 = v24 - 1;
      v27 = (v24 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v28 = *(_QWORD *)(v25 + 8LL * v27);
      if ( v11 != v28 )
      {
        v29 = 1;
        while ( v28 != -4096 )
        {
          a5 = (unsigned int)(v29 + 1);
          v27 = v26 & (v29 + v27);
          v28 = *(_QWORD *)(v25 + 8LL * v27);
          if ( v11 == v28 )
            goto LABEL_15;
          ++v29;
        }
        goto LABEL_30;
      }
LABEL_15:
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v10 != v9 );
    v7 = v8;
LABEL_17:
    v7 += 8;
  }
  while ( v58 != v7 );
  v18 = v62;
  v19 = &v62[8 * (unsigned int)v63];
  if ( v62 != v19 )
  {
    do
    {
      v20 = *((_QWORD *)v19 - 1);
      if ( *(_QWORD *)(v20 + 16) )
      {
        v21 = sub_ACADE0(*(__int64 ***)(v20 + 8));
        sub_BD84D0(v20, v21);
      }
      v19 -= 8;
      sub_B43D60((_QWORD *)v20);
    }
    while ( v18 != v19 );
    v19 = v62;
  }
  if ( v19 != v64 )
    _libc_free((unsigned __int64)v19);
}
