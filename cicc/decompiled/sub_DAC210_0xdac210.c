// Function: sub_DAC210
// Address: 0xdac210
//
__int64 __fastcall sub_DAC210(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r13
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 *v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // r14
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // ecx
  __int64 v22; // rsi
  int v23; // ecx
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r11
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rdi
  _QWORD *v36; // rsi
  _QWORD *v37; // r12
  _QWORD *v38; // rax
  _QWORD *v39; // r14
  _QWORD *v40; // rax
  _QWORD *v41; // rdi
  __int64 v42; // rsi
  __int64 result; // rax
  int v44; // eax
  int v45; // edi
  int v46; // ecx
  int v47; // r11d
  __int64 v48; // [rsp+18h] [rbp-318h]
  __int64 v49; // [rsp+28h] [rbp-308h]
  _QWORD *v50; // [rsp+28h] [rbp-308h]
  _QWORD *v51; // [rsp+30h] [rbp-300h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-2F8h]
  unsigned int v53; // [rsp+3Ch] [rbp-2F4h]
  _QWORD v54[16]; // [rsp+40h] [rbp-2F0h] BYREF
  _BYTE *v55; // [rsp+C0h] [rbp-270h] BYREF
  __int64 v56; // [rsp+C8h] [rbp-268h]
  _BYTE v57[128]; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v58; // [rsp+150h] [rbp-1E0h] BYREF
  __int64 *v59; // [rsp+158h] [rbp-1D8h]
  __int64 v60; // [rsp+160h] [rbp-1D0h]
  int v61; // [rsp+168h] [rbp-1C8h]
  char v62; // [rsp+16Ch] [rbp-1C4h]
  char v63; // [rsp+170h] [rbp-1C0h] BYREF
  __int64 *v64; // [rsp+1F0h] [rbp-140h] BYREF
  __int64 v65; // [rsp+1F8h] [rbp-138h]
  _BYTE v66[304]; // [rsp+200h] [rbp-130h] BYREF

  v3 = a2;
  v51 = v54;
  v64 = (__int64 *)v66;
  v54[0] = a2;
  v65 = 0x2000000000LL;
  v59 = (__int64 *)&v63;
  v53 = 16;
  v58 = 0;
  v60 = 16;
  v61 = 0;
  v62 = 1;
  v55 = v57;
  v56 = 0x1000000000LL;
  v4 = 1;
  while ( 1 )
  {
    v52 = v4 - 1;
    sub_D9A290(a1, v3, 0);
    sub_D9A290(a1, v3, 1u);
    if ( *(_DWORD *)(a1 + 1208) )
    {
      v35 = *(unsigned int *)(a1 + 1216);
      v36 = *(_QWORD **)(a1 + 1200);
      v37 = &v36[8 * v35];
      if ( v36 != v37 )
      {
        v38 = *(_QWORD **)(a1 + 1200);
        while ( 1 )
        {
          while ( 1 )
          {
            v39 = v38;
            if ( *v38 != -4096 )
              break;
            if ( v38[1] != -4096 )
              goto LABEL_39;
            v38 += 8;
            if ( v37 == v38 )
              goto LABEL_3;
          }
          if ( *v38 != -8192 || v38[1] != -8192 )
            break;
          v38 += 8;
          if ( v37 == v38 )
            goto LABEL_3;
        }
LABEL_39:
        if ( v37 != v38 )
        {
          do
          {
            v40 = v39 + 8;
            if ( v39[1] == v3 )
            {
              while ( v37 != v40 )
              {
                if ( *v40 == -4096 )
                {
                  if ( v40[1] != -4096 )
                    break;
                  v40 += 8;
                }
                else
                {
                  if ( *v40 != -8192 || v40[1] != -8192 )
                    break;
                  v40 += 8;
                }
              }
              v41 = (_QWORD *)v39[3];
              if ( v41 != v39 + 5 )
              {
                v50 = v40;
                _libc_free(v41, v39 + 5);
                v40 = v50;
              }
              *v39 = -8192;
              v39[1] = -8192;
              v39 = v40;
              v36 = *(_QWORD **)(a1 + 1200);
              --*(_DWORD *)(a1 + 1208);
              v35 = *(unsigned int *)(a1 + 1216);
              ++*(_DWORD *)(a1 + 1212);
            }
            else
            {
              while ( v37 != v40 )
              {
                if ( *v40 == -4096 )
                {
                  if ( v40[1] != -4096 )
                    goto LABEL_44;
                  v40 += 8;
                }
                else
                {
                  if ( *v40 != -8192 || v40[1] != -8192 )
                  {
LABEL_44:
                    v39 = v40;
                    goto LABEL_45;
                  }
                  v40 += 8;
                }
              }
              v39 = v37;
            }
LABEL_45:
            ;
          }
          while ( v39 != &v36[8 * v35] );
        }
      }
    }
LABEL_3:
    v5 = *(unsigned int *)(a1 + 1184);
    v6 = *(_QWORD *)(a1 + 1168);
    if ( (_DWORD)v5 )
    {
      v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (__int64 *)(v6 + 56LL * v7);
      v9 = *v8;
      if ( *v8 == v3 )
      {
LABEL_5:
        if ( v8 != (__int64 *)(v6 + 56 * v5) )
          sub_D927B0((__int64)&v55, &v55[8 * (unsigned int)v56], v8[1], v8[1] + 8LL * *((unsigned int *)v8 + 4));
      }
      else
      {
        v46 = 1;
        while ( v9 != -4096 )
        {
          v47 = v46 + 1;
          v7 = (v5 - 1) & (v46 + v7);
          v8 = (__int64 *)(v6 + 56LL * v7);
          v9 = *v8;
          if ( *v8 == v3 )
            goto LABEL_5;
          v46 = v47;
        }
      }
    }
    v10 = sub_AA5930(**(_QWORD **)(v3 + 32));
    v15 = v11;
    v16 = v10;
    if ( (__int64 *)v10 != v11 )
    {
      while ( 1 )
      {
        if ( v62 )
        {
          v17 = v59;
          v12 = HIDWORD(v60);
          v11 = &v59[HIDWORD(v60)];
          if ( v59 != v11 )
          {
            while ( v16 != *v17 )
            {
              if ( v11 == ++v17 )
                goto LABEL_33;
            }
            goto LABEL_13;
          }
LABEL_33:
          if ( HIDWORD(v60) < (unsigned int)v60 )
            break;
        }
        sub_C8CC70((__int64)&v58, v16, (__int64)v11, v12, v13, v14);
        if ( (_BYTE)v11 )
          goto LABEL_30;
LABEL_13:
        if ( !v16 )
          BUG();
        v18 = *(_QWORD *)(v16 + 32);
        if ( !v18 )
          BUG();
        v16 = 0;
        if ( *(_BYTE *)(v18 - 24) == 84 )
          v16 = v18 - 24;
        if ( v15 == (__int64 *)v16 )
          goto LABEL_18;
      }
      ++HIDWORD(v60);
      *v11 = v16;
      ++v58;
LABEL_30:
      v33 = (unsigned int)v65;
      v12 = HIDWORD(v65);
      v34 = (unsigned int)v65 + 1LL;
      if ( v34 > HIDWORD(v65) )
      {
        sub_C8D5F0((__int64)&v64, v66, v34, 8u, v13, v14);
        v33 = (unsigned int)v65;
      }
      v11 = v64;
      v64[v33] = v16;
      LODWORD(v65) = v65 + 1;
      goto LABEL_13;
    }
LABEL_18:
    sub_D988C0(a1, (__int64)&v64, (__int64)&v58, (__int64)&v55);
    v21 = *(_DWORD *)(a1 + 896);
    v22 = *(_QWORD *)(a1 + 880);
    if ( v21 )
    {
      v23 = v21 - 1;
      v24 = v23 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v25 = (__int64 *)(v22 + 16LL * v24);
      v19 = *v25;
      if ( *v25 == v3 )
      {
LABEL_20:
        *v25 = -8192;
        --*(_DWORD *)(a1 + 888);
        ++*(_DWORD *)(a1 + 892);
      }
      else
      {
        v44 = 1;
        while ( v19 != -4096 )
        {
          v45 = v44 + 1;
          v24 = v23 & (v44 + v24);
          v25 = (__int64 *)(v22 + 16LL * v24);
          v19 = *v25;
          if ( *v25 == v3 )
            goto LABEL_20;
          v44 = v45;
        }
      }
    }
    v26 = *(_QWORD *)(v3 + 8);
    v27 = v52;
    v28 = *(_QWORD *)(v3 + 16) - v26;
    v29 = v28 >> 3;
    if ( (v28 >> 3) + (unsigned __int64)v52 > v53 )
    {
      v48 = v28;
      v49 = v26;
      sub_C8D5F0((__int64)&v51, v54, v29 + v52, 8u, v19, v20);
      v27 = v52;
      v28 = v48;
      v26 = v49;
    }
    v30 = (__int64)&v51[v27];
    if ( v28 > 0 )
    {
      v31 = 0;
      do
      {
        *(_QWORD *)(v30 + 8 * v31) = *(_QWORD *)(v26 + 8 * v31);
        ++v31;
      }
      while ( v29 - v31 > 0 );
      v27 = v52;
    }
    v32 = v29 + v27;
    v52 = v29 + v27;
    v4 = v29 + v27;
    if ( !((_DWORD)v29 + (_DWORD)v27) )
      break;
    v3 = v51[(unsigned int)v32 - 1];
  }
  v42 = (__int64)v55;
  result = sub_DAB940(a1, (__int64)v55, (unsigned int)v56, v30, v32, v20);
  if ( v55 != v57 )
    result = _libc_free(v55, v42);
  if ( !v62 )
    result = _libc_free(v59, v42);
  if ( v64 != (__int64 *)v66 )
    result = _libc_free(v64, v42);
  if ( v51 != v54 )
    return _libc_free(v51, v42);
  return result;
}
