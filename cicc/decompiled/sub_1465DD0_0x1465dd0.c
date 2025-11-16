// Function: sub_1465DD0
// Address: 0x1465dd0
//
void __fastcall sub_1465DD0(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r14
  _BYTE *v6; // r13
  _BYTE *v7; // rcx
  int v8; // eax
  char v9; // dl
  __int64 v10; // rdi
  _BYTE *v11; // rdx
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 *v14; // rcx
  __int64 *v15; // rsi
  int v16; // eax
  int v17; // esi
  __int64 v18; // rcx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdi
  int v23; // eax
  int v24; // esi
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  int v29; // eax
  int v30; // r9d
  int v31; // eax
  int v32; // r9d
  __int64 v33; // [rsp+10h] [rbp-130h] BYREF
  __int64 *v34; // [rsp+18h] [rbp-128h]
  __int64 *v35; // [rsp+20h] [rbp-120h]
  __int64 v36; // [rsp+28h] [rbp-118h]
  int v37; // [rsp+30h] [rbp-110h]
  _BYTE v38[72]; // [rsp+38h] [rbp-108h] BYREF
  _BYTE *v39; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+88h] [rbp-B8h]
  _BYTE v41[176]; // [rsp+90h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v39 = v41;
  v3 = *(_QWORD *)(v2 + 8);
  v40 = 0x1000000000LL;
  if ( v3 )
  {
    v4 = v3;
    v5 = 0;
    do
    {
      v4 = *(_QWORD *)(v4 + 8);
      ++v5;
    }
    while ( v4 );
    v6 = v41;
    if ( v5 > 16 )
    {
      sub_16CD150(&v39, v41, v5, 8);
      v6 = &v39[8 * (unsigned int)v40];
    }
    do
    {
      v6 += 8;
      *((_QWORD *)v6 - 1) = sub_1648700(v3);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v3 );
    v7 = v39;
    v8 = v5 + v40;
  }
  else
  {
    v7 = v41;
    v8 = 0;
  }
  LODWORD(v40) = v8;
  v33 = 0;
  v34 = (__int64 *)v38;
  v35 = (__int64 *)v38;
  v36 = 8;
  v37 = 0;
LABEL_13:
  v11 = &v7[8 * v8];
  while ( v8 )
  {
    v12 = *((_QWORD *)v11 - 1);
    --v8;
    v11 -= 8;
    LODWORD(v40) = v8;
    if ( v2 != v12 )
    {
      v13 = v34;
      if ( v35 != v34 )
        goto LABEL_9;
      v14 = &v34[HIDWORD(v36)];
      if ( v34 != v14 )
      {
        v15 = 0;
        while ( v12 != *v13 )
        {
          if ( *v13 == -2 )
          {
            v15 = v13;
            if ( v13 + 1 == v14 )
              goto LABEL_23;
            ++v13;
          }
          else if ( v14 == ++v13 )
          {
            if ( !v15 )
              goto LABEL_38;
LABEL_23:
            *v15 = v12;
            v10 = *(_QWORD *)(a1 + 32);
            --v37;
            ++v33;
            if ( *(_BYTE *)(v12 + 16) != 77 )
              goto LABEL_11;
            goto LABEL_24;
          }
        }
        goto LABEL_12;
      }
LABEL_38:
      if ( HIDWORD(v36) < (unsigned int)v36 )
      {
        ++HIDWORD(v36);
        *v14 = v12;
        ++v33;
LABEL_10:
        v10 = *(_QWORD *)(a1 + 32);
        if ( *(_BYTE *)(v12 + 16) == 77 )
        {
LABEL_24:
          v16 = *(_DWORD *)(v10 + 616);
          if ( v16 )
          {
            v17 = v16 - 1;
            v18 = *(_QWORD *)(v10 + 600);
            v19 = (v16 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v20 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v12 == *v20 )
            {
LABEL_26:
              *v20 = -16;
              --*(_DWORD *)(v10 + 608);
              ++*(_DWORD *)(v10 + 612);
              v10 = *(_QWORD *)(a1 + 32);
            }
            else
            {
              v29 = 1;
              while ( v21 != -8 )
              {
                v30 = v29 + 1;
                v19 = v17 & (v29 + v19);
                v20 = (__int64 *)(v18 + 16LL * v19);
                v21 = *v20;
                if ( v12 == *v20 )
                  goto LABEL_26;
                v29 = v30;
              }
            }
          }
        }
LABEL_11:
        sub_1464220(v10, v12);
        sub_145C820((__int64 *)&v39, &v39[8 * (unsigned int)v40], *(_QWORD *)(v12 + 8), 0);
      }
      else
      {
LABEL_9:
        sub_16CCBA0(&v33, v12);
        if ( v9 )
          goto LABEL_10;
      }
LABEL_12:
      v7 = v39;
      v8 = v40;
      goto LABEL_13;
    }
  }
  v22 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(v2 + 16) == 77 )
  {
    v23 = *(_DWORD *)(v22 + 616);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v22 + 600);
      v26 = (v23 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v2 == *v27 )
      {
LABEL_37:
        *v27 = -16;
        --*(_DWORD *)(v22 + 608);
        ++*(_DWORD *)(v22 + 612);
        v22 = *(_QWORD *)(a1 + 32);
      }
      else
      {
        v31 = 1;
        while ( v28 != -8 )
        {
          v32 = v31 + 1;
          v26 = v24 & (v31 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v2 == *v27 )
            goto LABEL_37;
          v31 = v32;
        }
      }
    }
  }
  sub_1464220(v22, v2);
  if ( v35 != v34 )
    _libc_free((unsigned __int64)v35);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
}
