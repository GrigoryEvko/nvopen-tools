// Function: sub_1C0C360
// Address: 0x1c0c360
//
__int64 __fastcall sub_1C0C360(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // r9d
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rcx
  __int64 v17; // r9
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // r8
  __int64 **v21; // rax
  __int64 **v22; // r15
  __int64 *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rdx
  int v28; // eax
  int v29; // r11d
  int v30; // eax
  int v31; // eax
  int v32; // r10d
  __int64 v33; // rax
  int v35; // [rsp+14h] [rbp-ACh]
  __int64 **v36; // [rsp+18h] [rbp-A8h]
  __int64 *v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+28h] [rbp-98h] BYREF
  __int64 v39; // [rsp+30h] [rbp-90h] BYREF
  __int64 v40; // [rsp+38h] [rbp-88h] BYREF
  __int64 v41[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v42; // [rsp+50h] [rbp-70h]
  __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h]
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 *v46; // [rsp+70h] [rbp-50h]
  __int64 *v47; // [rsp+78h] [rbp-48h]
  __int64 v48; // [rsp+80h] [rbp-40h]
  __int64 **v49; // [rsp+88h] [rbp-38h]

  v38 = a2;
  v41[0] = 0;
  v41[1] = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  sub_1C08D60(v41, 0);
  v6 = v46;
  if ( v46 == (__int64 *)(v48 - 8) )
  {
    sub_1B4ECC0(v41, &v38);
    v7 = v38;
  }
  else
  {
    v7 = v38;
    if ( v46 )
    {
      *v46 = v38;
      v6 = v46;
    }
    v46 = v6 + 1;
  }
  v8 = *(unsigned int *)(a1 + 64);
  v40 = v7;
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 48);
    v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == v7 )
    {
LABEL_7:
      if ( v11 != (__int64 *)(v9 + 16 * v8) )
      {
        v35 = *(_DWORD *)(sub_1C0A020(a1 + 40, &v40)[1] + 16);
        goto LABEL_9;
      }
    }
    else
    {
      v31 = 1;
      while ( v12 != -8 )
      {
        v32 = v31 + 1;
        v33 = ((_DWORD)v8 - 1) & (v10 + v31);
        v10 = v33;
        v11 = (__int64 *)(v9 + 16 * v33);
        v12 = *v11;
        if ( *v11 == v7 )
          goto LABEL_7;
        v31 = v32;
      }
    }
  }
  v35 = 7;
LABEL_9:
  v13 = v46;
  while ( v42 != v46 )
  {
LABEL_10:
    if ( v47 == v13 )
    {
      v39 = (*(v49 - 1))[63];
      j_j___libc_free_0(v13, 512);
      v15 = *(_DWORD *)(a4 + 24);
      v26 = (__int64)(*--v49 + 64);
      v47 = *v49;
      v48 = v26;
      v46 = v47 + 63;
      if ( !v15 )
      {
LABEL_33:
        ++*(_QWORD *)a4;
        goto LABEL_34;
      }
    }
    else
    {
      v14 = *(v13 - 1);
      v15 = *(_DWORD *)(a4 + 24);
      v46 = v13 - 1;
      v39 = v14;
      if ( !v15 )
        goto LABEL_33;
    }
    v16 = v39;
    v17 = *(_QWORD *)(a4 + 8);
    v18 = (v15 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v19 = v17 + 40LL * v18;
    v20 = *(_QWORD *)v19;
    if ( v39 == *(_QWORD *)v19 )
    {
LABEL_13:
      v21 = *(__int64 ***)(v19 + 16);
      v22 = &v21[*(unsigned int *)(v19 + 32)];
      if ( *(_DWORD *)(v19 + 24) && v22 != v21 )
      {
        while ( *v21 == (__int64 *)-8LL || *v21 == (__int64 *)-16LL )
        {
          if ( v22 == ++v21 )
            goto LABEL_18;
        }
        if ( v22 != v21 )
        {
          while ( 2 )
          {
            v24 = *v21;
            if ( !*((_DWORD *)*v21 + 3) && v35 )
            {
              v36 = v21;
              v25 = *v24;
              v37 = *v21;
              *((_DWORD *)v24 + 3) = v35;
              v40 = v25;
              sub_1C09F00(a3, &v40);
              v40 = *v37;
              sub_1C09F00(v41, &v40);
              v21 = v36;
            }
            if ( ++v21 != v22 )
            {
              while ( *v21 == (__int64 *)-8LL || *v21 == (__int64 *)-16LL )
              {
                if ( v22 == ++v21 )
                {
                  v13 = v46;
                  if ( v42 != v46 )
                    goto LABEL_10;
                  return sub_1C08CE0(v41);
                }
              }
              if ( v22 != v21 )
                continue;
            }
            break;
          }
        }
      }
      goto LABEL_18;
    }
    v29 = 1;
    v27 = 0;
    while ( v20 != -8 )
    {
      if ( v20 == -16 && !v27 )
        v27 = v19;
      v18 = (v15 - 1) & (v29 + v18);
      v19 = v17 + 40LL * v18;
      v20 = *(_QWORD *)v19;
      if ( v39 == *(_QWORD *)v19 )
        goto LABEL_13;
      ++v29;
    }
    v30 = *(_DWORD *)(a4 + 16);
    if ( !v27 )
      v27 = v19;
    ++*(_QWORD *)a4;
    v28 = v30 + 1;
    if ( 4 * v28 < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a4 + 20) - v28 > v15 >> 3 )
        goto LABEL_42;
      goto LABEL_35;
    }
LABEL_34:
    v15 *= 2;
LABEL_35:
    sub_1C0BC70(a4, v15);
    sub_1C09AC0(a4, &v39, &v40);
    v27 = v40;
    v16 = v39;
    v28 = *(_DWORD *)(a4 + 16) + 1;
LABEL_42:
    *(_DWORD *)(a4 + 16) = v28;
    if ( *(_QWORD *)v27 != -8 )
      --*(_DWORD *)(a4 + 20);
    *(_QWORD *)v27 = v16;
    *(_QWORD *)(v27 + 8) = 0;
    *(_QWORD *)(v27 + 16) = 0;
    *(_QWORD *)(v27 + 24) = 0;
    *(_DWORD *)(v27 + 32) = 0;
LABEL_18:
    v13 = v46;
  }
  return sub_1C08CE0(v41);
}
