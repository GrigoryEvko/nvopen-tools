// Function: sub_39B6010
// Address: 0x39b6010
//
__int64 __fastcall sub_39B6010(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 *v5; // r12
  unsigned __int8 v6; // bl
  __int64 v7; // r13
  unsigned int v8; // r14d
  unsigned int v9; // eax
  char v10; // al
  unsigned __int8 v11; // al
  _QWORD *v12; // r13
  int v13; // ebx
  __int64 v14; // r15
  unsigned int v15; // eax
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  char v18; // dl
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rcx
  int v22; // ebx
  __int64 v23; // rcx
  int v24; // r15d
  __int64 *v25; // r14
  int v26; // r12d
  __int64 v27; // rdx
  int v28; // ebx
  int v29; // r13d
  int v30; // r13d
  int v31; // ebx
  __int64 v32; // rcx
  __int64 *v33; // r13
  int v34; // r12d
  int v35; // r14d
  __int64 v36; // rdx
  unsigned int v38; // [rsp+Ch] [rbp-74h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  unsigned int v40; // [rsp+18h] [rbp-68h]
  int v41; // [rsp+1Ch] [rbp-64h]
  int v42; // [rsp+20h] [rbp-60h]
  int v43; // [rsp+24h] [rbp-5Ch]
  __int64 *v44; // [rsp+28h] [rbp-58h]
  __int64 v45; // [rsp+30h] [rbp-50h]
  int v46; // [rsp+38h] [rbp-48h]
  unsigned int v47; // [rsp+3Ch] [rbp-44h]
  int v48; // [rsp+40h] [rbp-40h]
  unsigned int v49; // [rsp+44h] [rbp-3Ch]
  int v50; // [rsp+44h] [rbp-3Ch]
  _QWORD *v51; // [rsp+48h] [rbp-38h]
  int v52; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v5 = a1;
  v6 = a4;
  v7 = *(_QWORD *)(a2 + 32);
  v51 = (_QWORD *)a3;
  v40 = -1;
  v8 = v7;
  v45 = **(_QWORD **)(a2 + 16);
  v44 = **(__int64 ***)(a3 + 16);
  if ( (_DWORD)v7 )
  {
    _BitScanReverse(&v9, v7);
    v40 = 31 - (v9 ^ 0x1F);
  }
  v10 = *(_BYTE *)(a2 + 8);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(v45 + 8);
  v48 = ((unsigned __int8)(v10 - 1) < 6u) + 51;
  v47 = 1;
  v11 = (sub_1F43D80(a1[2], *a1, a2, a4) >> 32) - 14;
  if ( v11 <= 0x5Fu )
    v47 = word_4533E80[v11];
  v46 = v6 + 1;
  if ( v47 < (unsigned int)v7 )
  {
    v41 = 0;
    v12 = (_QWORD *)a2;
    v43 = 0;
    v42 = 0;
    while ( 1 )
    {
      v14 = v5[2];
      v8 >>= 1;
      v43 += v46;
      v15 = sub_1F43D70(v14, v48);
      v16 = v15;
      if ( v15 == 134 && *((_BYTE *)v51 + 8) == 16 )
        v16 = 135;
      v49 = v16;
      v17 = sub_1F43D80(v14, *v5, (__int64)v12, v16);
      v18 = *((_BYTE *)v12 + 8);
      v13 = v17;
      v19 = HIDWORD(v17);
      if ( v18 == 16 )
      {
        if ( (unsigned __int8)(v19 - 14) > 0x5Fu
          || (v20 = (unsigned __int8)v19, !*(_QWORD *)(v14 + 8LL * (unsigned __int8)v19 + 120)) )
        {
LABEL_21:
          v21 = 0;
          v39 = v12[4];
          if ( v51 )
          {
            v21 = (__int64)v51;
            if ( *((_BYTE *)v51 + 8) == 16 )
              v21 = *(_QWORD *)v51[2];
          }
          v22 = 0;
          v50 = sub_39B5490(v5, v48, *(_QWORD *)v12[2], v21, 0);
          if ( (int)v12[4] > 0 )
          {
            v38 = v8;
            v24 = 0;
            v25 = v5;
            v26 = v12[4];
            do
            {
              v27 = (__int64)v12;
              if ( *((_BYTE *)v12 + 8) == 16 )
                v27 = *(_QWORD *)v12[2];
              ++v24;
              v22 += sub_1F43D80(v25[2], *v25, v27, v23);
            }
            while ( v26 != v24 );
            v5 = v25;
            v8 = v38;
          }
          v13 = v39 * v50 + v22;
          goto LABEL_11;
        }
      }
      else
      {
        if ( !(_BYTE)v19 )
          goto LABEL_10;
        v20 = (unsigned __int8)v19;
        if ( !*(_QWORD *)(v14 + 8LL * (unsigned __int8)v19 + 120) )
          goto LABEL_10;
      }
      if ( v49 > 0x102 || *(_BYTE *)(v49 + v14 + 259 * v20 + 2422) != 2 )
        goto LABEL_11;
      if ( v18 == 16 )
        goto LABEL_21;
LABEL_10:
      v13 = 1;
LABEL_11:
      v42 += sub_39B5490(v5, 0x37u, (__int64)v12, (__int64)v51, 0) + v13;
      v12 = sub_16463B0((__int64 *)v45, v8);
      ++v41;
      v51 = sub_16463B0(v44, v8);
      if ( v47 >= v8 )
      {
        v40 -= v41;
        v4 = (__int64)v12;
        v28 = v42 + v43;
        goto LABEL_34;
      }
    }
  }
  v28 = 0;
LABEL_34:
  v29 = sub_39B5490(v5, v48, v4, (__int64)v51, 0);
  v30 = v28 + v40 * (v46 + sub_39B5490(v5, 0x37u, v4, (__int64)v51, 0) + v29);
  v31 = 0;
  if ( (int)*(_QWORD *)(v4 + 32) > 0 )
  {
    v32 = 0;
    v52 = v30;
    v33 = v5;
    v34 = *(_QWORD *)(v4 + 32);
    v35 = 0;
    do
    {
      v36 = v4;
      if ( *(_BYTE *)(v4 + 8) == 16 )
        v36 = **(_QWORD **)(v4 + 16);
      ++v35;
      v31 += sub_1F43D80(v33[2], *v33, v36, v32);
    }
    while ( v34 != v35 );
    v5 = v33;
    v30 = v52;
    v31 *= 3;
  }
  return v30 + v31 + (unsigned int)sub_39B5490(v5, 0x37u, v45, (__int64)v44, 0);
}
