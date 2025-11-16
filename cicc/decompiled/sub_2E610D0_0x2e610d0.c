// Function: sub_2E610D0
// Address: 0x2e610d0
//
void __fastcall sub_2E610D0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r15
  int v3; // r14d
  __int64 v4; // r12
  unsigned int v6; // eax
  int v7; // r10d
  __int64 v8; // r9
  unsigned int v9; // r8d
  __int64 v10; // rdx
  __int64 v11; // rdi
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // eax
  int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r10d
  __int64 v22; // r11
  int v23; // edx
  int v24; // eax
  unsigned __int64 v25; // rdx
  __int64 v26; // r13
  int v27; // eax
  _DWORD *v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  unsigned int v35; // r13d
  __int64 v36; // r8
  __int64 v37; // rsi
  int v38; // r11d
  int i; // [rsp+8h] [rbp-D8h]
  unsigned int v40; // [rsp+Ch] [rbp-D4h]
  __int64 v41; // [rsp+10h] [rbp-D0h]
  _BYTE *v42; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+38h] [rbp-A8h]
  _BYTE v44[32]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v45; // [rsp+60h] [rbp-80h] BYREF
  __int64 v46; // [rsp+68h] [rbp-78h]
  _QWORD v47[14]; // [rsp+70h] [rbp-70h] BYREF

  v2 = v47;
  v3 = 0;
  v4 = a2;
  v42 = v44;
  v43 = 0x800000000LL;
  v46 = 0x800000001LL;
  v41 = a1 + 8;
  v6 = 1;
  v45 = v47;
  v47[0] = a2;
  while ( 2 )
  {
    v13 = *(unsigned int *)(a1 + 32);
    if ( !(_DWORD)v13 )
    {
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_9;
    }
    v7 = v13 - 1;
    v8 = *(_QWORD *)(a1 + 16);
    v9 = (v13 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = v8 + 16LL * v9;
    v11 = *(_QWORD *)v10;
    if ( v4 == *(_QWORD *)v10 )
    {
      v12 = v43;
      if ( v6 == *(_DWORD *)&v42[4 * (unsigned int)v43 - 4] )
        goto LABEL_23;
      goto LABEL_4;
    }
    v40 = (v13 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v22 = *(_QWORD *)v10;
    v19 = 0;
    for ( i = 1; ; ++i )
    {
      if ( v22 == -4096 )
      {
        v24 = *(_DWORD *)(a1 + 24);
        if ( !v19 )
          v19 = v10;
        ++*(_QWORD *)(a1 + 8);
        v18 = v24 + 1;
        if ( 4 * (v24 + 1) < (unsigned int)(3 * v13) )
        {
          if ( (int)v13 - *(_DWORD *)(a1 + 28) - v18 > (unsigned int)v13 >> 3 )
          {
LABEL_35:
            *(_DWORD *)(a1 + 24) = v18;
            if ( *(_QWORD *)v19 != -4096 )
              --*(_DWORD *)(a1 + 28);
            ++v3;
            *(_QWORD *)v19 = v4;
            *(_DWORD *)(v19 + 8) = v3;
            v25 = (unsigned int)v43;
            *(_DWORD *)(v19 + 12) = 0;
            v26 = (unsigned int)v46;
            v27 = v25;
            if ( v25 >= HIDWORD(v43) )
            {
              if ( HIDWORD(v43) < v25 + 1 )
              {
                sub_C8D5F0((__int64)&v42, v44, v25 + 1, 4u, v25 + 1, v8);
                v25 = (unsigned int)v43;
              }
              *(_DWORD *)&v42[4 * v25] = v26;
              v26 = (unsigned int)v46;
              LODWORD(v43) = v43 + 1;
            }
            else
            {
              v28 = &v42[4 * v25];
              if ( v28 )
              {
                *v28 = v46;
                v27 = v43;
                v26 = (unsigned int)v46;
              }
              LODWORD(v43) = v27 + 1;
            }
            sub_2E5D970(
              (__int64)&v45,
              (char *)&v45[v26],
              *(char **)(v4 + 112),
              (char *)(*(_QWORD *)(v4 + 112) + 8LL * *(unsigned int *)(v4 + 120)));
            v31 = *(unsigned int *)(a1 + 48);
            if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v31 + 1, 8u, v29, v30);
              v31 = *(unsigned int *)(a1 + 48);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v31) = v4;
            v6 = v46;
            ++*(_DWORD *)(a1 + 48);
            v2 = v45;
            goto LABEL_5;
          }
          sub_2E60EF0(v41, v13);
          v32 = *(_DWORD *)(a1 + 32);
          if ( v32 )
          {
            v33 = v32 - 1;
            v34 = *(_QWORD *)(a1 + 16);
            v8 = 1;
            v35 = v33 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v36 = 0;
            v18 = *(_DWORD *)(a1 + 24) + 1;
            v19 = v34 + 16LL * v35;
            v37 = *(_QWORD *)v19;
            if ( v4 != *(_QWORD *)v19 )
            {
              while ( v37 != -4096 )
              {
                if ( !v36 && v37 == -8192 )
                  v36 = v19;
                v35 = v33 & (v8 + v35);
                v19 = v34 + 16LL * v35;
                v37 = *(_QWORD *)v19;
                if ( v4 == *(_QWORD *)v19 )
                  goto LABEL_35;
                v8 = (unsigned int)(v8 + 1);
              }
              if ( v36 )
                v19 = v36;
            }
            goto LABEL_35;
          }
LABEL_67:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
LABEL_9:
        sub_2E60EF0(v41, 2 * v13);
        v14 = *(_DWORD *)(a1 + 32);
        if ( v14 )
        {
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 16);
          v17 = (v14 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v18 = *(_DWORD *)(a1 + 24) + 1;
          v19 = v16 + 16LL * v17;
          v20 = *(_QWORD *)v19;
          if ( *(_QWORD *)v19 != v4 )
          {
            v21 = 1;
            v8 = 0;
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v8 )
                v8 = v19;
              v17 = v15 & (v21 + v17);
              v19 = v16 + 16LL * v17;
              v20 = *(_QWORD *)v19;
              if ( v4 == *(_QWORD *)v19 )
                goto LABEL_35;
              ++v21;
            }
            if ( v8 )
              v19 = v8;
          }
          goto LABEL_35;
        }
        goto LABEL_67;
      }
      if ( v19 || v22 != -8192 )
        v10 = v19;
      v40 = v7 & (v40 + i);
      v22 = *(_QWORD *)(v8 + 16LL * v40);
      if ( v4 == v22 )
        break;
      v19 = v10;
      v10 = v8 + 16LL * v40;
    }
    v12 = v43;
    if ( *(_DWORD *)&v42[4 * (unsigned int)v43 - 4] == v6 )
    {
      v23 = 1;
      while ( v11 != -4096 )
      {
        v38 = v23 + 1;
        v9 = v7 & (v23 + v9);
        v10 = v8 + 16LL * v9;
        v11 = *(_QWORD *)v10;
        if ( v4 == *(_QWORD *)v10 )
          goto LABEL_23;
        v23 = v38;
      }
      v10 = v8 + 16 * v13;
LABEL_23:
      *(_DWORD *)(v10 + 12) = v3;
      LODWORD(v43) = v12 - 1;
    }
LABEL_4:
    LODWORD(v46) = --v6;
LABEL_5:
    if ( v6 )
    {
      v4 = v2[v6 - 1];
      continue;
    }
    break;
  }
  if ( v2 != v47 )
    _libc_free((unsigned __int64)v2);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
}
