// Function: sub_2798350
// Address: 0x2798350
//
void __fastcall sub_2798350(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r12
  int v4; // r14d
  __int64 v5; // rax
  int v6; // r8d
  unsigned int v7; // esi
  int v8; // eax
  __int64 v9; // r13
  __int64 v10; // r15
  int v11; // ecx
  __int64 v12; // rax
  int *v13; // rax
  __int64 v14; // r9
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // edx
  int v19; // eax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdi
  __int64 v23; // r9
  int v24; // r11d
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // eax
  __int64 v28; // rdi
  int v29; // r11d
  unsigned int v30; // ecx
  __int64 v31; // rsi
  __int64 v32; // [rsp+8h] [rbp-C8h]
  int v33; // [rsp+1Ch] [rbp-B4h]
  int v34; // [rsp+1Ch] [rbp-B4h]
  int v35; // [rsp+1Ch] [rbp-B4h]
  int v36; // [rsp+1Ch] [rbp-B4h]
  int v37; // [rsp+1Ch] [rbp-B4h]
  _BYTE *v38; // [rsp+28h] [rbp-A8h]
  _QWORD v39[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+40h] [rbp-90h]
  _BYTE *v41; // [rsp+50h] [rbp-80h] BYREF
  __int64 v42; // [rsp+58h] [rbp-78h]
  _BYTE v43[112]; // [rsp+60h] [rbp-70h] BYREF

  v32 = a1 + 728;
  sub_278E0A0(a1 + 728);
  v41 = v43;
  v42 = 0x800000000LL;
  sub_2797FE0((__int64)&v41, a2);
  v3 = &v41[8 * (unsigned int)v42];
  v38 = v41;
  if ( v41 == v3 )
    goto LABEL_27;
  v4 = 1;
  do
  {
    v5 = *((_QWORD *)v3 - 1);
    v6 = v4++;
    v39[0] = 0;
    v39[1] = 0;
    v40 = v5;
    if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
    {
      v33 = v6;
      sub_BD73F0((__int64)v39);
      v6 = v33;
    }
    v7 = *(_DWORD *)(a1 + 752);
    if ( !v7 )
    {
      ++*(_QWORD *)(a1 + 728);
LABEL_8:
      v34 = v6;
      sub_27965B0(v32, 2 * v7);
      v8 = *(_DWORD *)(a1 + 752);
      v6 = v34;
      if ( v8 )
      {
        v9 = v40;
        v27 = v8 - 1;
        v28 = *(_QWORD *)(a1 + 736);
        v23 = 0;
        v29 = 1;
        v30 = v27 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v10 = v28 + 32LL * v30;
        v31 = *(_QWORD *)(v10 + 16);
        if ( v40 != v31 )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v23 )
              v23 = v10;
            v30 = v27 & (v29 + v30);
            v10 = v28 + 32LL * v30;
            v31 = *(_QWORD *)(v10 + 16);
            if ( v40 == v31 )
              goto LABEL_10;
            ++v29;
          }
          goto LABEL_48;
        }
        goto LABEL_10;
      }
      goto LABEL_9;
    }
    v9 = v40;
    v14 = *(_QWORD *)(a1 + 736);
    v15 = (v7 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v16 = v14 + 32LL * v15;
    v17 = *(_QWORD *)(v16 + 16);
    if ( v40 == v17 )
    {
LABEL_21:
      v13 = (int *)(v16 + 24);
      goto LABEL_22;
    }
    v18 = 1;
    v10 = 0;
    while ( v17 != -4096 )
    {
      if ( !v10 && v17 == -8192 )
        v10 = v16;
      v15 = (v7 - 1) & (v18 + v15);
      v16 = v14 + 32LL * v15;
      v17 = *(_QWORD *)(v16 + 16);
      if ( v40 == v17 )
        goto LABEL_21;
      ++v18;
    }
    if ( !v10 )
      v10 = v16;
    v19 = *(_DWORD *)(a1 + 744);
    ++*(_QWORD *)(a1 + 728);
    v11 = v19 + 1;
    if ( 4 * (v19 + 1) >= 3 * v7 )
      goto LABEL_8;
    if ( v7 - *(_DWORD *)(a1 + 748) - v11 <= v7 >> 3 )
    {
      v37 = v6;
      sub_27965B0(v32, v7);
      v20 = *(_DWORD *)(a1 + 752);
      v6 = v37;
      if ( v20 )
      {
        v9 = v40;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a1 + 736);
        v23 = 0;
        v24 = 1;
        LODWORD(v25) = v21 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v10 = v22 + 32LL * (unsigned int)v25;
        v26 = *(_QWORD *)(v10 + 16);
        if ( v40 != v26 )
        {
          while ( v26 != -4096 )
          {
            if ( v26 == -8192 && !v23 )
              v23 = v10;
            v25 = v21 & (unsigned int)(v25 + v24);
            v10 = v22 + 32 * v25;
            v26 = *(_QWORD *)(v10 + 16);
            if ( v40 == v26 )
              goto LABEL_10;
            ++v24;
          }
LABEL_48:
          if ( v23 )
            v10 = v23;
        }
LABEL_10:
        v11 = *(_DWORD *)(a1 + 744) + 1;
        goto LABEL_11;
      }
LABEL_9:
      v9 = v40;
      v10 = 0;
      goto LABEL_10;
    }
LABEL_11:
    *(_DWORD *)(a1 + 744) = v11;
    if ( *(_QWORD *)(v10 + 16) == -4096 )
    {
      if ( v9 != -4096 )
        goto LABEL_16;
    }
    else
    {
      --*(_DWORD *)(a1 + 748);
      v12 = *(_QWORD *)(v10 + 16);
      if ( v9 != v12 )
      {
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
        {
          v35 = v6;
          sub_BD60C0((_QWORD *)v10);
          v6 = v35;
        }
LABEL_16:
        *(_QWORD *)(v10 + 16) = v9;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        {
          v36 = v6;
          sub_BD73F0(v10);
          v6 = v36;
        }
      }
    }
    *(_DWORD *)(v10 + 24) = 0;
    v13 = (int *)(v10 + 24);
LABEL_22:
    *v13 = v6;
    if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
      sub_BD60C0(v39);
    v3 -= 8;
  }
  while ( v38 != v3 );
  v3 = v41;
LABEL_27:
  *(_BYTE *)(a1 + 760) = 0;
  if ( v3 != v43 )
    _libc_free((unsigned __int64)v3);
}
