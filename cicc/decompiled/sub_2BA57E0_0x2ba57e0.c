// Function: sub_2BA57E0
// Address: 0x2ba57e0
//
char __fastcall sub_2BA57E0(__int64 *a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r8d
  unsigned int v15; // esi
  __int64 *v16; // rdx
  __int64 v17; // r10
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // rdx
  int v25; // edi
  __int64 v26; // r8
  int v27; // edi
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // r10
  __int64 v31; // rcx
  __int64 v32; // rax
  int v33; // r8d
  int v34; // edi
  char v35; // r12
  __int64 v36; // rax
  char v37; // r12
  int v38; // ecx
  int v39; // edx
  unsigned int v40; // edx
  __int64 v41; // r9
  __int64 v42; // rsi
  int v43; // ecx
  int v44; // edi
  unsigned int v45; // ecx
  __int64 *v46; // r8
  __int64 v47; // r10
  int v48; // r8d
  int v49; // r11d
  int v50; // edx
  int v51; // r11d
  int v52; // eax
  int v53; // r9d
  __int64 v55[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *a1;
  v9 = *(_QWORD *)(v8 + 168);
  if ( v9 == a1[1] )
  {
    v35 = a2;
    if ( a3 )
      sub_2BA36A0(v8, a3, 1, a1[2], a5, a6);
    if ( !a2 )
      goto LABEL_36;
  }
  else
  {
    v10 = *(_QWORD *)(v8 + 160);
    v11 = v8;
    if ( v9 != v10 )
    {
      do
      {
        v12 = *(_QWORD *)(v10 + 40);
        if ( v12 == *(_QWORD *)v8 )
        {
          v13 = *(_DWORD *)(v8 + 104);
          a6 = *(_QWORD *)(v8 + 88);
          if ( v13 )
          {
            v14 = v13 - 1;
            v15 = v14 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v16 = (__int64 *)(a6 + 16LL * v15);
            v17 = *v16;
            if ( *v16 == v10 )
            {
LABEL_6:
              v18 = v16[1];
              if ( v18 && *(_DWORD *)(v18 + 136) == *(_DWORD *)(v8 + 204) )
              {
                *(_QWORD *)(v18 + 144) = -1;
                *(_DWORD *)(v18 + 48) = 0;
                *(_DWORD *)(v18 + 96) = 0;
                v12 = *(_QWORD *)(v10 + 40);
                v8 = *a1;
              }
            }
            else
            {
              v50 = 1;
              while ( v17 != -4096 )
              {
                v51 = v50 + 1;
                v15 = v14 & (v50 + v15);
                v16 = (__int64 *)(a6 + 16LL * v15);
                v17 = *v16;
                if ( *v16 == v10 )
                  goto LABEL_6;
                v50 = v51;
              }
            }
          }
        }
        v19 = *(_QWORD *)(v10 + 32);
        if ( v19 == v12 + 48 || !v19 )
          v10 = 0;
        else
          v10 = v19 - 24;
        v11 = v8;
      }
      while ( *(_QWORD *)(v8 + 168) != v10 );
    }
    if ( !a3 )
      goto LABEL_16;
    sub_2BA36A0(v8, a3, 1, a1[2], v11, a6);
  }
  v11 = *a1;
LABEL_16:
  sub_2B2F0D0(v11);
  v20 = *a1;
  v21 = *(_QWORD *)(*a1 + 160);
  v22 = *a1 + 112;
  if ( *(_QWORD *)(*a1 + 168) != v21 )
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(v21 + 40);
      if ( *(_QWORD *)v20 == v24 )
      {
        v25 = *(_DWORD *)(v20 + 104);
        v26 = *(_QWORD *)(v20 + 88);
        if ( v25 )
        {
          v27 = v25 - 1;
          v28 = v27 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( *v29 == v21 )
          {
LABEL_24:
            v31 = v29[1];
            if ( v31 )
            {
              if ( *(_DWORD *)(v31 + 136) == *(_DWORD *)(v20 + 204) )
              {
                v55[0] = v29[1];
                if ( *(_QWORD *)(v31 + 16) == v31 && *(_DWORD *)(v31 + 144) != -1 )
                {
                  v32 = v31;
                  v33 = 0;
                  while ( 1 )
                  {
                    v34 = *(_DWORD *)(v32 + 148);
                    if ( v34 == -1 )
                      break;
                    v32 = *(_QWORD *)(v32 + 24);
                    v33 += v34;
                    if ( !v32 )
                    {
                      if ( v33 || *(_BYTE *)(v31 + 152) )
                        break;
                      sub_2BA3420(v22, v55);
                      v24 = *(_QWORD *)(v21 + 40);
                      v23 = *(_QWORD *)(v21 + 32);
                      if ( v23 )
                        goto LABEL_19;
                      goto LABEL_34;
                    }
                  }
                }
              }
            }
          }
          else
          {
            v52 = 1;
            while ( v30 != -4096 )
            {
              v53 = v52 + 1;
              v28 = v27 & (v52 + v28);
              v29 = (__int64 *)(v26 + 16LL * v28);
              v30 = *v29;
              if ( *v29 == v21 )
                goto LABEL_24;
              v52 = v53;
            }
          }
        }
      }
      v23 = *(_QWORD *)(v21 + 32);
      if ( !v23 )
        goto LABEL_34;
LABEL_19:
      if ( v23 == v24 + 48 )
      {
LABEL_34:
        v21 = 0;
        if ( !*(_QWORD *)(v20 + 168) )
          break;
      }
      else
      {
        v21 = v23 - 24;
        if ( *(_QWORD *)(v20 + 168) == v21 )
          break;
      }
    }
  }
  v35 = 1;
LABEL_36:
  LOBYTE(v36) = a3 == 0;
  v37 = (a3 == 0) & v35;
  while ( 1 )
  {
    if ( !v37 )
    {
      if ( !a3 )
        return v36;
      v36 = a3;
      v38 = 0;
      while ( 1 )
      {
        v39 = *(_DWORD *)(v36 + 148);
        if ( v39 == -1 )
          break;
        v36 = *(_QWORD *)(v36 + 24);
        v38 += v39;
        if ( !v36 )
        {
          if ( v38 || *(_BYTE *)(a3 + 152) )
            break;
          return v36;
        }
      }
    }
    v36 = *a1;
    v40 = *(_DWORD *)(*a1 + 152);
    if ( !v40 )
      return v36;
    v41 = *(_QWORD *)(v36 + 120);
    v42 = *(_QWORD *)(*(_QWORD *)(v36 + 144) + 8LL * v40 - 8);
    v43 = *(_DWORD *)(v36 + 136);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = (v43 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
      v46 = (__int64 *)(v41 + 8LL * v45);
      v47 = *v46;
      if ( v42 == *v46 )
      {
LABEL_49:
        *v46 = -8192;
        v40 = *(_DWORD *)(v36 + 152);
        --*(_DWORD *)(v36 + 128);
        ++*(_DWORD *)(v36 + 132);
      }
      else
      {
        v48 = 1;
        while ( v47 != -4096 )
        {
          v49 = v48 + 1;
          v45 = v44 & (v48 + v45);
          v46 = (__int64 *)(v41 + 8LL * v45);
          v47 = *v46;
          if ( v42 == *v46 )
            goto LABEL_49;
          v48 = v49;
        }
      }
    }
    *(_DWORD *)(v36 + 152) = v40 - 1;
    LOBYTE(v36) = sub_2BA4E90(*a1, v42, *a1 + 112);
  }
}
