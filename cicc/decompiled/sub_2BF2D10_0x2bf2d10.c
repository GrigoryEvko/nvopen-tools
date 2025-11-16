// Function: sub_2BF2D10
// Address: 0x2bf2d10
//
void __fastcall sub_2BF2D10(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 *v5; // r12
  __int64 v6; // r9
  int v7; // esi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 *v11; // r15
  unsigned __int64 v12; // rax
  _QWORD *v13; // r10
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r8
  char v18; // cl
  unsigned int v19; // esi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 *v23; // rsi
  _QWORD *v24; // rax
  _QWORD *v25; // r10
  unsigned __int16 v26; // r8
  unsigned int v27; // eax
  int v28; // edx
  unsigned int v29; // edi
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  int v34; // r11d
  __int64 *v35; // r15
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 *v38; // [rsp+18h] [rbp-68h]
  _QWORD *v39; // [rsp+18h] [rbp-68h]
  _QWORD *v40; // [rsp+18h] [rbp-68h]
  _QWORD *v41; // [rsp+20h] [rbp-60h]
  unsigned __int16 v42; // [rsp+20h] [rbp-60h]
  _QWORD *v43; // [rsp+20h] [rbp-60h]
  __int64 *v44; // [rsp+20h] [rbp-60h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+20h] [rbp-60h]
  __int64 *v47; // [rsp+28h] [rbp-58h]
  __int64 v48; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v49; // [rsp+38h] [rbp-48h] BYREF
  __int64 *v50; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v51; // [rsp+48h] [rbp-38h]

  v36 = a2 + 120;
  v50 = a1;
  v3 = *sub_2BF2B80(a2 + 120, (__int64 *)&v50);
  v4 = sub_2BF0570((__int64)a1);
  v5 = *(__int64 **)(v4 + 56);
  v47 = &v5[*(unsigned int *)(v4 + 64)];
  while ( v47 != v5 )
  {
    v48 = sub_2BF0520(*v5);
    v17 = sub_2BF0540(v48);
    v18 = *(_BYTE *)(a2 + 128) & 1;
    if ( v18 )
    {
      v6 = a2 + 136;
      v7 = 3;
    }
    else
    {
      v19 = *(_DWORD *)(a2 + 144);
      v6 = *(_QWORD *)(a2 + 136);
      if ( !v19 )
      {
        v27 = *(_DWORD *)(a2 + 128);
        ++*(_QWORD *)(a2 + 120);
        v50 = 0;
        v28 = (v27 >> 1) + 1;
LABEL_43:
        v29 = 3 * v19;
        goto LABEL_44;
      }
      v7 = v19 - 1;
    }
    v8 = v7 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v48 == *v9 )
    {
LABEL_5:
      v11 = (__int64 *)v9[1];
      goto LABEL_6;
    }
    v34 = 1;
    v35 = 0;
    while ( v10 != -4096 )
    {
      if ( v10 == -8192 && !v35 )
        v35 = v9;
      v8 = v7 & (v34 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v48 == *v9 )
        goto LABEL_5;
      ++v34;
    }
    v29 = 12;
    v19 = 4;
    if ( !v35 )
      v35 = v9;
    v27 = *(_DWORD *)(a2 + 128);
    ++*(_QWORD *)(a2 + 120);
    v50 = v35;
    v28 = (v27 >> 1) + 1;
    if ( !v18 )
    {
      v19 = *(_DWORD *)(a2 + 144);
      goto LABEL_43;
    }
LABEL_44:
    if ( 4 * v28 >= v29 )
    {
      v46 = v17;
      v19 *= 2;
LABEL_64:
      sub_2ACA3E0(v36, v19);
      sub_2ABFB80(v36, &v48, &v50);
      v27 = *(_DWORD *)(a2 + 128);
      v17 = v46;
      goto LABEL_46;
    }
    if ( v19 - *(_DWORD *)(a2 + 132) - v28 <= v19 >> 3 )
    {
      v46 = v17;
      goto LABEL_64;
    }
LABEL_46:
    *(_DWORD *)(a2 + 128) = (2 * (v27 >> 1) + 2) | v27 & 1;
    v30 = v50;
    if ( *v50 != -4096 )
      --*(_DWORD *)(a2 + 132);
    v31 = v48;
    v30[1] = 0;
    v11 = 0;
    *v30 = v31;
LABEL_6:
    v12 = v11[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v12 == v11 + 6 )
      goto LABEL_31;
    if ( !v12 )
      BUG();
    v13 = (_QWORD *)(v12 - 24);
    v14 = *(unsigned __int8 *)(v12 - 24);
    if ( (unsigned int)(v14 - 30) > 0xA )
LABEL_31:
      BUG();
    if ( (_BYTE)v14 != 31 )
    {
      if ( (_BYTE)v14 == 36 )
      {
        v23 = *(__int64 **)(v12 + 24);
        v49 = v23;
        if ( v23 )
        {
          v41 = (_QWORD *)(v12 - 24);
          sub_B96E90((__int64)&v49, (__int64)v23, 1);
          v13 = v41;
        }
        sub_B43D60(v13);
        sub_B43C20((__int64)&v50, (__int64)v11);
        v38 = v50;
        v42 = v51;
        v24 = sub_BD2C40(72, 1u);
        v25 = v24;
        if ( v24 )
        {
          v26 = v42;
          v43 = v24;
          sub_B4C8F0((__int64)v24, v3, 1u, (__int64)v38, v26);
          v25 = v43;
        }
        v17 = (__int64)(v25 + 6);
        v50 = v49;
        if ( v49 )
        {
          v44 = v25 + 6;
          v39 = v25;
          sub_B96E90((__int64)&v50, (__int64)v49, 1);
          v17 = (__int64)v44;
          if ( v44 == (__int64 *)&v50 )
          {
            if ( v50 )
              sub_B91220((__int64)&v50, (__int64)v50);
            goto LABEL_40;
          }
          v25 = v39;
          v32 = v39[6];
          if ( !v32 )
          {
LABEL_52:
            v33 = (unsigned __int8 *)v50;
            v25[6] = v50;
            if ( v33 )
              sub_B976B0((__int64)&v50, v33, v17);
            goto LABEL_40;
          }
        }
        else if ( (__int64 **)v17 == &v50 || (v32 = v25[6]) == 0 )
        {
LABEL_40:
          if ( v49 )
            sub_B91220((__int64)&v49, (__int64)v49);
          goto LABEL_18;
        }
        v40 = v25;
        v45 = v17;
        sub_B91220(v17, v32);
        v25 = v40;
        v17 = v45;
        goto LABEL_52;
      }
      v13 = 0;
LABEL_24:
      v20 = &v13[-4 * (**(_QWORD **)(v17 + 80) != (_QWORD)a1) - 4];
      if ( *v20 )
      {
        v14 = v20[2];
        v21 = v20[1];
        *(_QWORD *)v14 = v21;
        if ( v21 )
        {
          v14 = v20[2];
          *(_QWORD *)(v21 + 16) = v14;
        }
      }
      *v20 = v3;
      if ( v3 )
      {
        v22 = *(_QWORD *)(v3 + 16);
        v14 = v3 + 16;
        v20[1] = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = v20 + 1;
        v20[2] = v14;
        *(_QWORD *)(v3 + 16) = v20;
      }
      goto LABEL_18;
    }
    if ( (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) == 3 )
      goto LABEL_24;
    if ( *(_QWORD *)(v12 - 56) )
    {
      v14 = *(_QWORD *)(v12 - 40);
      v15 = *(_QWORD *)(v12 - 48);
      *(_QWORD *)v14 = v15;
      if ( v15 )
      {
        v14 = *(_QWORD *)(v12 - 40);
        *(_QWORD *)(v15 + 16) = v14;
      }
    }
    *(_QWORD *)(v12 - 56) = v3;
    if ( v3 )
    {
      v16 = *(_QWORD *)(v3 + 16);
      v14 = v3 + 16;
      *(_QWORD *)(v12 - 48) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = v12 - 48;
      *(_QWORD *)(v12 - 40) = v14;
      *(_QWORD *)(v3 + 16) = v12 - 56;
    }
LABEL_18:
    v50 = v11;
    ++v5;
    v51 = v3 & 0xFFFFFFFFFFFFFFFBLL;
    sub_FFB3D0(a2 + 200, (unsigned __int64 *)&v50, 1, v14, v17, v6);
  }
}
