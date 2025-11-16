// Function: sub_18D3860
// Address: 0x18d3860
//
__int64 __fastcall sub_18D3860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // r12d
  __int64 v7; // r14
  _QWORD *v8; // r11
  _QWORD *v9; // r15
  __int64 v10; // r13
  __int64 i; // rdi
  int v13; // edi
  __int64 v14; // rax
  unsigned __int8 v15; // al
  int v16; // eax
  __int64 v17; // r15
  int v18; // eax
  unsigned int v19; // ecx
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *j; // rdx
  _QWORD *v23; // r13
  _QWORD *v24; // r12
  _QWORD *v25; // rbx
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rdi
  int v29; // edi
  __int64 v30; // rax
  unsigned __int8 v31; // al
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // ecx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rdx
  _QWORD *v39; // rdi
  unsigned int v40; // eax
  int v41; // eax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  int v44; // ebx
  __int64 v45; // r12
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *k; // rdx
  _QWORD *v49; // rax
  unsigned __int8 v50; // [rsp+Fh] [rbp-61h]
  __int64 v52; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  _QWORD *v57; // [rsp+28h] [rbp-48h]
  __int64 v58[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = sub_14399D0(a2);
  switch ( v6 )
  {
    case 0u:
    case 1u:
      for ( i = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            ;
            i = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)) )
      {
        v14 = sub_1649C60(i);
        v13 = 23;
        v7 = v14;
        v15 = *(_BYTE *)(v14 + 16);
        if ( v15 <= 0x17u )
          goto LABEL_12;
        if ( v15 != 78 )
          break;
        v13 = 21;
        if ( *(_BYTE *)(*(_QWORD *)(v7 - 24) + 16LL) )
          goto LABEL_12;
        v16 = sub_1438F00(*(_QWORD *)(v7 - 24));
        if ( !(unsigned __int8)sub_1439C90(v16) )
        {
LABEL_18:
          v58[0] = v7;
          v17 = sub_18D2B00(a5 + 64, v58);
          v50 = sub_18DBCD0(v17);
          if ( v50 )
          {
            if ( v6 != 1 )
            {
              v58[0] = a2;
              v34 = sub_18D3440(a4, v58);
              *(_BYTE *)v34 = *(_BYTE *)(v17 + 8);
              *(_BYTE *)(v34 + 1) = *(_BYTE *)(v17 + 9);
              v38 = *(_QWORD *)(v17 + 16);
              *(_QWORD *)(v34 + 8) = v38;
              if ( v17 + 24 != v34 + 16 )
              {
                v54 = v34;
                sub_16CCD50(v34 + 16, v17 + 24, v38, v35, v36, v37);
                v34 = v54;
              }
              if ( v17 + 80 != v34 + 72 )
              {
                v55 = v34;
                sub_16CCD50(v34 + 72, v17 + 80, v38, v35, v36, v37);
                v34 = v55;
              }
              *(_BYTE *)(v34 + 128) = *(_BYTE *)(v17 + 136);
            }
            sub_18DB9D0(v17, 0);
            v50 = 0;
          }
          goto LABEL_3;
        }
LABEL_13:
        ;
      }
      v13 = 2 * (v15 != 29) + 21;
LABEL_12:
      if ( !(unsigned __int8)sub_1439C90(v13) )
        goto LABEL_18;
      goto LABEL_13;
    case 4u:
      v28 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      while ( 2 )
      {
        v30 = sub_1649C60(v28);
        v29 = 23;
        v7 = v30;
        v31 = *(_BYTE *)(v30 + 16);
        if ( v31 <= 0x17u )
          goto LABEL_38;
        if ( v31 != 78 )
        {
          v29 = 2 * (v31 != 29) + 21;
          goto LABEL_38;
        }
        v29 = 21;
        if ( *(_BYTE *)(*(_QWORD *)(v7 - 24) + 16LL) )
        {
LABEL_38:
          if ( !(unsigned __int8)sub_1439C90(v29) )
            goto LABEL_44;
LABEL_39:
          v28 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
          continue;
        }
        break;
      }
      v32 = sub_1438F00(*(_QWORD *)(v7 - 24));
      if ( (unsigned __int8)sub_1439C90(v32) )
        goto LABEL_39;
LABEL_44:
      v58[0] = v7;
      v33 = sub_18D2B00(a5 + 64, v58);
      v50 = sub_18DBB50(v33, a1 + 312, a2);
LABEL_3:
      v8 = *(_QWORD **)(a5 + 96);
      v57 = *(_QWORD **)(a5 + 104);
      if ( v57 != v8 )
      {
        v9 = v8;
        v10 = a1 + 160;
        do
        {
          if ( v7 != *v9 )
          {
            v52 = *v9;
            if ( !(unsigned __int8)sub_18DBD70(v9 + 1, a2, *v9, v10, v6) )
              sub_18DBDD0(v9 + 1, a3, a2, v52, v10, v6);
          }
          v9 += 19;
        }
        while ( v57 != v9 );
      }
      return v50;
    case 7u:
    case 0x18u:
      return 0;
    case 8u:
      v18 = *(_DWORD *)(a5 + 80);
      ++*(_QWORD *)(a5 + 64);
      if ( !v18 )
      {
        if ( !*(_DWORD *)(a5 + 84) )
          goto LABEL_28;
        v20 = *(unsigned int *)(a5 + 88);
        if ( (unsigned int)v20 > 0x40 )
        {
          j___libc_free_0(*(_QWORD *)(a5 + 72));
          *(_QWORD *)(a5 + 72) = 0;
          *(_QWORD *)(a5 + 80) = 0;
          *(_DWORD *)(a5 + 88) = 0;
          goto LABEL_28;
        }
        goto LABEL_25;
      }
      v19 = 4 * v18;
      v20 = *(unsigned int *)(a5 + 88);
      if ( (unsigned int)(4 * v18) < 0x40 )
        v19 = 64;
      if ( (unsigned int)v20 <= v19 )
      {
LABEL_25:
        v21 = *(_QWORD **)(a5 + 72);
        for ( j = &v21[2 * v20]; j != v21; v21 += 2 )
          *v21 = -8;
        *(_QWORD *)(a5 + 80) = 0;
        goto LABEL_28;
      }
      v39 = *(_QWORD **)(a5 + 72);
      v40 = v18 - 1;
      if ( v40 )
      {
        _BitScanReverse(&v40, v40);
        v41 = 1 << (33 - (v40 ^ 0x1F));
        if ( v41 < 64 )
          v41 = 64;
        if ( (_DWORD)v20 == v41 )
        {
          *(_QWORD *)(a5 + 80) = 0;
          v49 = &v39[2 * (unsigned int)v20];
          do
          {
            if ( v39 )
              *v39 = -8;
            v39 += 2;
          }
          while ( v49 != v39 );
          goto LABEL_28;
        }
        v42 = (4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1);
        v43 = ((v42 | (v42 >> 2)) >> 4) | v42 | (v42 >> 2) | ((((v42 | (v42 >> 2)) >> 4) | v42 | (v42 >> 2)) >> 8);
        v44 = (v43 | (v43 >> 16)) + 1;
        v45 = 16 * ((v43 | (v43 >> 16)) + 1);
      }
      else
      {
        v45 = 2048;
        v44 = 128;
      }
      j___libc_free_0(v39);
      *(_DWORD *)(a5 + 88) = v44;
      v46 = (_QWORD *)sub_22077B0(v45);
      v47 = *(unsigned int *)(a5 + 88);
      *(_QWORD *)(a5 + 80) = 0;
      *(_QWORD *)(a5 + 72) = v46;
      for ( k = &v46[2 * v47]; k != v46; v46 += 2 )
      {
        if ( v46 )
          *v46 = -8;
      }
LABEL_28:
      v50 = 0;
      v23 = *(_QWORD **)(a5 + 96);
      v24 = *(_QWORD **)(a5 + 104);
      if ( v23 != v24 )
      {
        v25 = *(_QWORD **)(a5 + 96);
        do
        {
          v26 = v25[13];
          if ( v26 != v25[12] )
            _libc_free(v26);
          v27 = v25[6];
          if ( v27 != v25[5] )
            _libc_free(v27);
          v25 += 19;
        }
        while ( v24 != v25 );
        v50 = 0;
        *(_QWORD *)(a5 + 104) = v23;
      }
      return v50;
    default:
      v50 = 0;
      v7 = 0;
      goto LABEL_3;
  }
}
