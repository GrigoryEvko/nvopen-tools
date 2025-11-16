// Function: sub_36F5F40
// Address: 0x36f5f40
//
__int64 __fastcall sub_36F5F40(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 *v5; // rbx
  __int64 *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 i; // r12
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 j; // rbx
  int v13; // edx
  unsigned int v14; // ecx
  int *v15; // rdi
  int v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // r12d
  __int64 v19; // rsi
  __int64 v21; // r12
  __int64 v22; // r12
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int64 v27; // rdx
  int v28; // r12d
  unsigned int v29; // edx
  unsigned int v30; // ecx
  int *v31; // rdi
  int v32; // r8d
  int v33; // r13d
  int v34; // r11d
  unsigned int v35; // r10d
  _DWORD *v36; // rdi
  _DWORD *v37; // rcx
  int v38; // r9d
  int v39; // edx
  int v40; // edi
  int v41; // r10d
  __int64 v42; // rax
  int v43; // r11d
  int v44; // edi
  _DWORD *v45; // rsi
  int v46; // edi
  int v47; // r9d
  int v48; // esi
  __int64 v49; // r8
  int v50; // edi
  _DWORD *v51; // rax
  __int64 v53; // [rsp+28h] [rbp-E8h]
  __int64 v54; // [rsp+28h] [rbp-E8h]
  __int64 v55; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v56; // [rsp+38h] [rbp-D8h]
  __int64 v57; // [rsp+40h] [rbp-D0h]
  unsigned int v58; // [rsp+48h] [rbp-C8h]
  __int64 *v59; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+58h] [rbp-B8h]
  _BYTE v61[176]; // [rsp+60h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a2 + 328);
  v59 = (__int64 *)v61;
  v60 = 0x1000000000LL;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v53 = a2 + 320;
  if ( v2 == a2 + 320 )
  {
    v17 = 0;
    v19 = 0;
    v18 = 0;
    goto LABEL_25;
  }
  do
  {
    v3 = *(_QWORD *)(v2 + 56);
    v4 = v2 + 48;
    if ( v3 != v2 + 48 )
    {
      while ( (unsigned int)*(unsigned __int16 *)(v3 + 68) - 3446 > 5 )
      {
LABEL_6:
        if ( (*(_BYTE *)v3 & 4) != 0 )
        {
          v3 = *(_QWORD *)(v3 + 8);
          if ( v4 == v3 )
            goto LABEL_8;
        }
        else
        {
          while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
            v3 = *(_QWORD *)(v3 + 8);
          v3 = *(_QWORD *)(v3 + 8);
          if ( v4 == v3 )
            goto LABEL_8;
        }
      }
      v21 = *(_QWORD *)(v3 + 32);
      v22 = v21 + 40LL * (unsigned int)sub_2E88FE0(v3);
      sub_2E88FE0(v3);
      v25 = (unsigned int)v60;
      v26 = *(_QWORD *)(v3 + 32);
      v27 = (unsigned int)v60 + 1LL;
      if ( v27 > HIDWORD(v60) )
      {
        sub_C8D5F0((__int64)&v59, v61, v27, 8u, v23, v24);
        v25 = (unsigned int)v60;
      }
      v59[v25] = v3;
      LODWORD(v60) = v60 + 1;
      v28 = *(_DWORD *)(v22 + 8);
      if ( v58 )
      {
        v29 = v58 - 1;
        v30 = (v58 - 1) & (37 * v28);
        v31 = (int *)(v56 + 8LL * v30);
        v32 = *v31;
        if ( v28 == *v31 )
        {
LABEL_35:
          if ( v31 != (int *)(v56 + 8LL * v58) )
            v28 = v31[1];
        }
        else
        {
          v46 = 1;
          while ( v32 != -1 )
          {
            v47 = v46 + 1;
            v30 = v29 & (v46 + v30);
            v31 = (int *)(v56 + 8LL * v30);
            v32 = *v31;
            if ( v28 == *v31 )
              goto LABEL_35;
            v46 = v47;
          }
        }
        v33 = *(_DWORD *)(v26 + 8);
        v34 = 1;
        v35 = v29 & (37 * v33);
        v36 = (_DWORD *)(v56 + 8LL * v35);
        v37 = 0;
        v38 = *v36;
        if ( *v36 == v33 )
          goto LABEL_6;
        while ( v38 != -1 )
        {
          if ( v38 != -2 || v37 )
            v36 = v37;
          v35 = v29 & (v34 + v35);
          v38 = *(_DWORD *)(v56 + 8LL * v35);
          if ( v33 == v38 )
            goto LABEL_6;
          ++v34;
          v37 = v36;
          v36 = (_DWORD *)(v56 + 8LL * v35);
        }
        if ( !v37 )
          v37 = v36;
        ++v55;
        v39 = v57 + 1;
        if ( 4 * ((int)v57 + 1) < 3 * v58 )
        {
          if ( v58 - HIDWORD(v57) - v39 <= v58 >> 3 )
          {
            sub_2FFACA0((__int64)&v55, v58);
            if ( !v58 )
            {
LABEL_88:
              LODWORD(v57) = v57 + 1;
              BUG();
            }
            v48 = 1;
            LODWORD(v49) = (v58 - 1) & (37 * v33);
            v37 = (_DWORD *)(v56 + 8LL * (unsigned int)v49);
            v50 = *v37;
            v39 = v57 + 1;
            v51 = 0;
            if ( *v37 != v33 )
            {
              while ( v50 != -1 )
              {
                if ( v50 == -2 && !v51 )
                  v51 = v37;
                v49 = (v58 - 1) & ((_DWORD)v49 + v48);
                v37 = (_DWORD *)(v56 + 8 * v49);
                v50 = *v37;
                if ( v33 == *v37 )
                  goto LABEL_43;
                ++v48;
              }
              if ( v51 )
                v37 = v51;
            }
          }
          goto LABEL_43;
        }
      }
      else
      {
        ++v55;
        v33 = *(_DWORD *)(v26 + 8);
      }
      sub_2FFACA0((__int64)&v55, 2 * v58);
      if ( !v58 )
        goto LABEL_88;
      LODWORD(v42) = (v58 - 1) & (37 * v33);
      v37 = (_DWORD *)(v56 + 8LL * (unsigned int)v42);
      v43 = *v37;
      v39 = v57 + 1;
      if ( *v37 != v33 )
      {
        v44 = 1;
        v45 = 0;
        while ( v43 != -1 )
        {
          if ( !v45 && v43 == -2 )
            v45 = v37;
          v42 = (v58 - 1) & ((_DWORD)v42 + v44);
          v37 = (_DWORD *)(v56 + 8 * v42);
          v43 = *v37;
          if ( *v37 == v33 )
            goto LABEL_43;
          ++v44;
        }
        if ( v45 )
          v37 = v45;
      }
LABEL_43:
      LODWORD(v57) = v39;
      if ( *v37 != -1 )
        --HIDWORD(v57);
      *v37 = v33;
      v37[1] = v28;
      goto LABEL_6;
    }
LABEL_8:
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v53 != v2 );
  v54 = v2;
  if ( (_DWORD)v60 )
  {
    v5 = v59;
    v6 = &v59[(unsigned int)v60];
    do
    {
      v7 = *v5++;
      sub_2E88E20(v7);
    }
    while ( v6 != v5 );
    v8 = *(_QWORD *)(a2 + 328);
    if ( v2 != v8 )
    {
      do
      {
        for ( i = *(_QWORD *)(v8 + 56); v8 + 48 != i; i = *(_QWORD *)(i + 8) )
        {
          v10 = *(_QWORD *)(i + 32);
          v11 = v10 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
          for ( j = v10 + 40LL * (unsigned int)sub_2E88FE0(i); v11 != j; j += 40 )
          {
            if ( !*(_BYTE *)j )
            {
              v13 = *(_DWORD *)(j + 8);
              if ( v58 )
              {
                v14 = (v58 - 1) & (37 * v13);
                v15 = (int *)(v56 + 8LL * v14);
                v16 = *v15;
                if ( v13 == *v15 )
                {
LABEL_18:
                  if ( v15 != (int *)(v56 + 8LL * v58) )
                    sub_2EAB0C0(j, v15[1]);
                }
                else
                {
                  v40 = 1;
                  while ( v16 != -1 )
                  {
                    v41 = v40 + 1;
                    v14 = (v58 - 1) & (v40 + v14);
                    v15 = (int *)(v56 + 8LL * v14);
                    v16 = *v15;
                    if ( v13 == *v15 )
                      goto LABEL_18;
                    v40 = v41;
                  }
                }
              }
            }
          }
          if ( (*(_BYTE *)i & 4) == 0 )
          {
            while ( (*(_BYTE *)(i + 44) & 8) != 0 )
              i = *(_QWORD *)(i + 8);
          }
        }
        v8 = *(_QWORD *)(v8 + 8);
      }
      while ( v54 != v8 );
    }
    v17 = v56;
    v18 = 1;
    v19 = 8LL * v58;
  }
  else
  {
    v17 = v56;
    v18 = 0;
    v19 = 8LL * v58;
  }
LABEL_25:
  sub_C7D6A0(v17, v19, 4);
  if ( v59 != (__int64 *)v61 )
    _libc_free((unsigned __int64)v59);
  return v18;
}
