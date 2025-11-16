// Function: sub_893FE0
// Address: 0x893fe0
//
__int64 *__fastcall sub_893FE0(__int64 a1, __int64 *a2, unsigned int a3)
{
  _BYTE *v4; // r14
  __int64 *v5; // r12
  _QWORD *v6; // r15
  _QWORD *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 *v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // r9
  __m128i *v22; // rax
  int v23; // eax
  char v24; // dl
  __int64 v25; // rcx
  __int64 v27; // rax
  int v28; // eax
  int v29; // eax
  __int64 v30; // [rsp-10h] [rbp-250h]
  __int64 v31; // [rsp-8h] [rbp-248h]
  __int64 v32; // [rsp+8h] [rbp-238h]
  __int64 v33; // [rsp+8h] [rbp-238h]
  char v35; // [rsp+10h] [rbp-230h]
  __int64 v36; // [rsp+10h] [rbp-230h]
  __int64 *v38; // [rsp+18h] [rbp-228h]
  __int64 *v39; // [rsp+18h] [rbp-228h]
  int v40; // [rsp+24h] [rbp-21Ch]
  __int64 v41; // [rsp+28h] [rbp-218h]
  _QWORD v42[66]; // [rsp+30h] [rbp-210h] BYREF

  v4 = *(_BYTE **)(a1 + 88);
  v40 = sub_8D0B70(a1);
  v5 = sub_87F3D0(a1);
  v32 = sub_892920(a1);
  v6 = *(_QWORD **)(v32 + 88);
  v7 = sub_7259C0(v4[264]);
  v41 = v7[21];
  *((_BYTE *)v7 + 177) |= 0x10u;
  v5[11] = (__int64)v7;
  sub_877D80((__int64)v7, v5);
  v8 = (__int64)v5;
  sub_877F10((__int64)v7, (__int64)v5, v9, v10, v11, v12);
  v15 = a3;
  v16 = a2;
  if ( (v4[160] & 2) != 0
    || (v4[266] & 1) != 0
    || (*((_BYTE *)v5 + 81) & 0x10) != 0 && (*(_BYTE *)(v5[8] + 177) & 0x20) != 0 )
  {
    *((_BYTE *)v7 + 177) |= 0x20u;
  }
  if ( a3 )
  {
    *((_BYTE *)v7 + 177) |= 0x20u;
  }
  else if ( dword_4F04C44 != -1
         || (v15 = (__int64)qword_4F04C68, v27 = qword_4F04C68[0] + 776LL * dword_4F04C64,
                                           (*(_BYTE *)(v27 + 6) & 6) != 0)
         || *(_BYTE *)(v27 + 4) == 12 )
  {
    v17 = (__int64)a2;
    v28 = sub_893F30(a2, (__int64)v5, v15, v13, (__int64)a2, v14);
    v16 = a2;
    if ( v28 )
    {
      *((_BYTE *)v7 + 177) |= 0x20u;
      goto LABEL_7;
    }
  }
  v8 = (__int64)v6;
  v17 = a1;
  v38 = v16;
  sub_890140(a1, v6, (__int64)v5, (__int64)v16, (__int64)v16, v14);
  v16 = v38;
LABEL_7:
  *(_QWORD *)(v41 + 168) = v16;
  v18 = *(_QWORD *)(a1 + 88);
  v19 = *(_QWORD *)(v18 + 88);
  if ( v19 && (*(_BYTE *)(v18 + 160) & 1) == 0 )
    v18 = *(_QWORD *)(v19 + 88);
  else
    v19 = a1;
  *(_QWORD *)(v41 + 160) = *(_QWORD *)(v18 + 104);
  if ( (*((_BYTE *)v5 + 81) & 0x10) != 0 )
    *((_BYTE *)v7 + 88) = (v4[265] >> 6) | v7[11] & 0xFC;
  if ( (v4[160] & 4) != 0 )
  {
    *((_BYTE *)v7 + 88) = v7[11] & 0x8F | 0x20;
  }
  else if ( dword_4D047F8
         && (v17 = (__int64)v16, v36 = v19, v39 = v16, v29 = sub_88DB10(v16), v16 = v39, v19 = v36, v29) )
  {
    *((_BYTE *)v7 + 88) = v7[11] & 0x8F | 0x10;
    *(_BYTE *)(v41 + 109) &= 0xF8u;
  }
  else
  {
    *((_BYTE *)v7 + 88) = (4 * v4[265]) & 0x70 | v7[11] & 0x8F;
  }
  v20 = *(_QWORD *)(*(_QWORD *)(v19 + 88) + 176LL);
  if ( v20 && (v20 = *(_QWORD *)(v20 + 88)) != 0 )
  {
    if ( *(_QWORD *)(v20 + 104) )
    {
      memset(v42, 0, 0x1D8u);
      v42[19] = v42;
      v42[3] = *(_QWORD *)&dword_4F063F8;
      if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v42[22]) |= 1u;
      v21 = 0;
      if ( (*((_BYTE *)v7 + 89) & 4) != 0 )
        v21 = *(_QWORD *)(v7[5] + 32LL);
      v22 = (__m128i *)sub_5CF220(
                         *(const __m128i **)(v20 + 104),
                         1,
                         v32,
                         **(_QWORD **)(*(_QWORD *)(v19 + 88) + 32LL),
                         (__int64)v16,
                         v21,
                         1,
                         0);
      sub_66A990(v22, (__int64)v7, (__int64)v42, 1, 0, 0);
      v8 = v30;
      v17 = v31;
      if ( v7[13] )
      {
        v8 = 6;
        v17 = (__int64)v42;
        sub_656C00((__int64)v42, 6, (__int64)v7, 0, 1);
      }
      v35 = 1;
    }
    else
    {
      v35 = 1;
    }
  }
  else
  {
    v35 = 0;
  }
  v23 = 1;
  v24 = *((_BYTE *)v7 + 177) & 0x20;
  if ( (char)v4[160] >= 0 )
    LOBYTE(v23) = v24 == 0;
  v25 = (unsigned int)(32 * v23);
  *((_BYTE *)v7 + 141) = (32 * v23) | *((_BYTE *)v7 + 141) & 0xDF;
  if ( v24 )
  {
    if ( (char)v4[160] < 0 )
    {
LABEL_30:
      sub_890050((__int64)v4, (__int64)v7);
      sub_7365B0((__int64)v7, -1);
      goto LABEL_31;
    }
  }
  else
  {
    v8 = (__int64)v5;
    v17 = 0x8000;
    sub_8756F0(0x8000, (__int64)v5, v5 + 6, 0);
    if ( (char)v4[160] < 0 || (*((_BYTE *)v7 + 177) & 0x20) == 0 )
      goto LABEL_30;
  }
  v33 = v5[12];
  *(_DWORD *)(v33 + 96) = sub_880E90(v17, v8, v33, v25, (__int64)v16, v14);
  v7[16] = 1;
  *((_DWORD *)v7 + 34) = 1;
  if ( dword_4F07590 )
  {
    sub_7365B0((__int64)v7, -(((v4[160] >> 1) ^ 1) & 1));
    if ( a1 != *(_QWORD *)&dword_4D04988 )
      goto LABEL_32;
LABEL_46:
    *(_BYTE *)(v41 + 110) |= 1u;
    goto LABEL_32;
  }
LABEL_31:
  if ( a1 == *(_QWORD *)&dword_4D04988 )
    goto LABEL_46;
LABEL_32:
  if ( dword_4F077BC && v35 )
    *((_BYTE *)v7 + 143) = *(_BYTE *)(v20 + 143) & 0x80 | *((_BYTE *)v7 + 143) & 0x7F;
  sub_8CCE20(v5, v4);
  if ( v40 )
    sub_8D0B10();
  return v5;
}
