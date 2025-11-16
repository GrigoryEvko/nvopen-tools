// Function: sub_39673A0
// Address: 0x39673a0
//
void __fastcall sub_39673A0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 *v11; // r14
  __int64 v12; // rcx
  __int64 v13; // r8
  int v14; // eax
  int v15; // r8d
  int v16; // r9d
  unsigned __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // rax
  char v20; // dl
  __int64 v21; // rax
  unsigned __int8 v22; // dl
  char *v23; // r14
  __int64 v24; // rbx
  char *v25; // r12
  unsigned __int64 v26; // rax
  char *v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // rdx
  char *v30; // rax
  char *v31; // rsi
  char *v32; // rsi
  int v33; // r11d
  __int64 *v34; // r10
  int v35; // eax
  int v36; // ecx
  __int64 v37; // [rsp+8h] [rbp-178h]
  __int64 v38; // [rsp+10h] [rbp-170h] BYREF
  __int64 *v39; // [rsp+18h] [rbp-168h] BYREF
  _QWORD v40[2]; // [rsp+20h] [rbp-160h] BYREF
  int v41; // [rsp+30h] [rbp-150h]
  char v42[8]; // [rsp+38h] [rbp-148h] BYREF
  __int64 v43; // [rsp+40h] [rbp-140h]
  unsigned __int64 v44; // [rsp+48h] [rbp-138h]
  __int64 v45; // [rsp+A0h] [rbp-E0h] BYREF
  _BYTE *v46; // [rsp+A8h] [rbp-D8h]
  _BYTE *v47; // [rsp+B0h] [rbp-D0h]
  __int64 v48; // [rsp+B8h] [rbp-C8h]
  int v49; // [rsp+C0h] [rbp-C0h]
  _BYTE v50[184]; // [rsp+C8h] [rbp-B8h] BYREF

  v46 = v50;
  v47 = v50;
  v4 = *a2;
  v48 = 16;
  v49 = 0;
  v5 = *(_QWORD *)(v4 + 8);
  v45 = 0;
  if ( v5 )
  {
    while ( 1 )
    {
      v38 = sub_3961CF0(a1, v5, 1);
      if ( !v38 )
        goto LABEL_12;
      sub_1412190((__int64)&v45, v38);
      if ( !v20 )
        goto LABEL_12;
      v21 = *a2;
      v22 = *(_BYTE *)(*a2 + 16);
      if ( v22 > 0x17u )
      {
        v6 = *(_QWORD *)(v21 + 40);
      }
      else
      {
        if ( v22 != 17 )
          BUG();
        v6 = *(_QWORD *)(*(_QWORD *)(v21 + 24) + 80LL);
        if ( v6 )
          v6 -= 24;
      }
      if ( v38 == v6 )
        goto LABEL_12;
      sub_3963E30((__int64)v40, a1, a2, v38, 1);
      v7 = *(_DWORD *)(a1 + 288);
      if ( !v7 )
        break;
      v8 = v38;
      v9 = *(_QWORD *)(a1 + 272);
      v10 = (v7 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v38 != *v11 )
      {
        v33 = 1;
        v34 = 0;
        while ( v12 != -8 )
        {
          if ( v12 == -16 && !v34 )
            v34 = v11;
          v10 = (v7 - 1) & (v33 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v38 == *v11 )
            goto LABEL_7;
          ++v33;
        }
        v35 = *(_DWORD *)(a1 + 280);
        if ( v34 )
          v11 = v34;
        ++*(_QWORD *)(a1 + 264);
        v36 = v35 + 1;
        if ( 4 * (v35 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a1 + 284) - v36 > v7 >> 3 )
          {
LABEL_38:
            *(_DWORD *)(a1 + 280) = v36;
            if ( *v11 != -8 )
              --*(_DWORD *)(a1 + 284);
            *v11 = v8;
            *((_DWORD *)v11 + 2) = 0;
            goto LABEL_7;
          }
LABEL_43:
          sub_13FEAC0(a1 + 264, v7);
          sub_13FDDE0(a1 + 264, &v38, &v39);
          v11 = v39;
          v8 = v38;
          v36 = *(_DWORD *)(a1 + 280) + 1;
          goto LABEL_38;
        }
LABEL_42:
        v7 *= 2;
        goto LABEL_43;
      }
LABEL_7:
      v13 = sub_22077B0(0xD8u);
      v14 = *((_DWORD *)v11 + 2);
      *(_QWORD *)(v13 + 16) = v38;
      *(_DWORD *)(v13 + 24) = v14;
      v37 = v13;
      *(_QWORD *)(v13 + 32) = v40[0];
      *(_QWORD *)(v13 + 40) = v40[1];
      *(_DWORD *)(v13 + 48) = v41;
      sub_16CCEE0((_QWORD *)(v13 + 56), v13 + 96, 8, (__int64)v42);
      *(_QWORD *)(v37 + 160) = v37 + 176;
      *(_QWORD *)(v37 + 168) = 0x400000000LL;
      *(_WORD *)(v37 + 208) = 0;
      sub_2208C80((_QWORD *)v37, a1 + 136);
      v17 = v44;
      ++*(_QWORD *)(a1 + 152);
      if ( v17 != v43 )
        _libc_free(v17);
      v18 = *(_QWORD *)(a1 + 144) + 16LL;
      v19 = *((unsigned int *)a2 + 14);
      if ( (unsigned int)v19 >= *((_DWORD *)a2 + 15) )
      {
        sub_16CD150((__int64)(a2 + 6), a2 + 8, 0, 8, v15, v16);
        v19 = *((unsigned int *)a2 + 14);
      }
      *(_QWORD *)(a2[6] + 8 * v19) = v18;
      ++*((_DWORD *)a2 + 14);
LABEL_12:
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_20;
    }
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_42;
  }
LABEL_20:
  v23 = (char *)a2[6];
  v24 = 8LL * *((unsigned int *)a2 + 14);
  v25 = &v23[v24];
  if ( &v23[v24] != v23 )
  {
    _BitScanReverse64(&v26, v24 >> 3);
    sub_3961050((char *)a2[6], (__int64 *)&v23[v24], 2LL * (int)(63 - (v26 ^ 0x3F)));
    if ( (unsigned __int64)v24 <= 0x80 )
    {
      sub_3960E30(v23, &v23[v24]);
    }
    else
    {
      v27 = v23 + 128;
      sub_3960E30(v23, v23 + 128);
      if ( v25 != v23 + 128 )
      {
        do
        {
          while ( 1 )
          {
            v28 = *(_QWORD *)v27;
            v29 = *((_QWORD *)v27 - 1);
            v30 = v27 - 8;
            if ( *(_DWORD *)(v29 + 8) > *(_DWORD *)(*(_QWORD *)v27 + 8LL) )
              break;
            v32 = v27;
            v27 += 8;
            *(_QWORD *)v32 = v28;
            if ( v25 == v27 )
              goto LABEL_26;
          }
          do
          {
            *((_QWORD *)v30 + 1) = v29;
            v31 = v30;
            v29 = *((_QWORD *)v30 - 1);
            v30 -= 8;
          }
          while ( *(_DWORD *)(v28 + 8) < *(_DWORD *)(v29 + 8) );
          v27 += 8;
          *(_QWORD *)v31 = v28;
        }
        while ( v25 != v27 );
      }
    }
  }
LABEL_26:
  if ( v47 != v46 )
    _libc_free((unsigned __int64)v47);
}
