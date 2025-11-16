// Function: sub_15DB440
// Address: 0x15db440
//
void __fastcall sub_15DB440(__int64 *a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax
  char v5; // r12
  unsigned int v6; // eax
  __int64 *v7; // rbx
  int v8; // r15d
  __int64 v9; // rdx
  int v10; // r15d
  char v11; // al
  __int64 *v12; // rsi
  unsigned int v13; // eax
  unsigned int v14; // r9d
  unsigned __int64 v15; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  unsigned int v18; // eax
  char v19; // dl
  __int64 v20; // rax
  _QWORD *v21; // r15
  _QWORD *v22; // r12
  _BYTE *v23; // r8
  _QWORD *v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // r15
  _QWORD *v27; // r14
  int v28; // eax
  __int64 v29; // r8
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // r15
  __int64 *v34; // rbx
  unsigned int v35; // r9d
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  char v38; // al
  __int64 *v39; // rsi
  unsigned int v40; // eax
  unsigned int v41; // esi
  __m128i *v42; // r13
  __int64 v43; // rbx
  __m128i *v44; // r12
  unsigned __int64 v45; // rax
  const __m128i *v46; // rbx
  const __m128i *v47; // rdi
  _QWORD *v48; // r12
  __int64 v49; // rax
  __int64 v50; // [rsp+10h] [rbp-E0h]
  __int64 v51; // [rsp+10h] [rbp-E0h]
  __int64 *v52; // [rsp+20h] [rbp-D0h]
  __int64 v53; // [rsp+20h] [rbp-D0h]
  __int64 v54; // [rsp+28h] [rbp-C8h]
  __int64 *v55; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v56; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v57; // [rsp+48h] [rbp-A8h]
  __int64 v58; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v59; // [rsp+58h] [rbp-98h]
  _QWORD *v60; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v61; // [rsp+68h] [rbp-88h]
  _BYTE v62[48]; // [rsp+C0h] [rbp-30h] BYREF

  v3 = &v60;
  v54 = a2;
  v58 = 0;
  v59 = 1;
  do
  {
    *(_QWORD *)v3 = -8;
    v3 += 24;
    *((_QWORD *)v3 - 2) = -8;
  }
  while ( v3 != v62 );
  v5 = v59 & 1;
  if ( (_DWORD)a2 )
  {
    ++v58;
    LODWORD(a2) = sub_1454B60(4 * (int)a2 / 3u + 1);
    v6 = 4;
    if ( v5 )
      goto LABEL_6;
    goto LABEL_5;
  }
  ++v58;
  if ( !v5 )
  {
LABEL_5:
    v6 = v61;
LABEL_6:
    if ( (unsigned int)a2 > v6 )
      sub_15D0B40((__int64)&v58, a2);
  }
  v52 = &a1[2 * v54];
  if ( a1 == v52 )
    goto LABEL_25;
  v50 = a3;
  v7 = a1;
  do
  {
    v9 = v7[1];
    v56 = *v7;
    v10 = -((v9 & 4) == 0);
    v57 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = sub_15D0A10((__int64)&v58, &v56, &v55);
    v12 = v55;
    v8 = (v10 & 2) - 1;
    if ( v11 )
    {
      v8 += *((_DWORD *)v55 + 4);
      goto LABEL_11;
    }
    ++v58;
    v13 = ((unsigned int)v59 >> 1) + 1;
    if ( (v59 & 1) != 0 )
    {
      v14 = 4;
      if ( 4 * v13 >= 0xC )
      {
LABEL_20:
        v16 = 2 * v14;
        goto LABEL_21;
      }
    }
    else
    {
      v14 = v61;
      if ( 4 * v13 >= 3 * v61 )
        goto LABEL_20;
    }
    if ( v14 - (v13 + HIDWORD(v59)) > v14 >> 3 )
      goto LABEL_16;
    v16 = v14;
LABEL_21:
    sub_15D0B40((__int64)&v58, v16);
    sub_15D0A10((__int64)&v58, &v56, &v55);
    v12 = v55;
    v13 = ((unsigned int)v59 >> 1) + 1;
LABEL_16:
    LODWORD(v59) = v59 & 1 | (2 * v13);
    if ( *v12 != -8 || v12[1] != -8 )
      --HIDWORD(v59);
    *v12 = v56;
    v15 = v57;
    *((_DWORD *)v12 + 4) = 0;
    v12[1] = v15;
LABEL_11:
    *((_DWORD *)v12 + 4) = v8;
    v7 += 2;
  }
  while ( v52 != v7 );
  a3 = v50;
LABEL_25:
  v17 = v59;
  *(_DWORD *)(a3 + 8) = 0;
  v18 = v17 >> 1;
  if ( *(_DWORD *)(a3 + 12) < v18 )
  {
    sub_16CD150(a3, a3 + 16, v18, 16);
    v19 = v59 & 1;
    if ( (unsigned int)v59 >> 1 )
      goto LABEL_27;
  }
  else
  {
    v19 = v59 & 1;
    if ( v18 )
    {
LABEL_27:
      if ( v19 )
      {
        v24 = v60;
        v22 = &v60;
        v23 = v62;
        if ( v60 != (_QWORD *)-8LL )
          goto LABEL_30;
        goto LABEL_48;
      }
      v20 = v61;
      v21 = v60;
      v22 = v60;
      v23 = &v60[3 * v61];
      if ( v23 != (_BYTE *)v60 )
      {
        while ( 1 )
        {
          v24 = (_QWORD *)*v22;
          if ( *v22 == -8 )
          {
LABEL_48:
            if ( v22[1] != -8 )
              goto LABEL_31;
          }
          else
          {
LABEL_30:
            if ( v24 != (_QWORD *)-16LL || v22[1] != -16 )
              goto LABEL_31;
          }
          v22 += 3;
          if ( v22 == (_QWORD *)v23 )
            goto LABEL_31;
        }
      }
LABEL_33:
      v25 = 3 * v20;
      goto LABEL_34;
    }
  }
  if ( v19 )
  {
    v48 = &v60;
    v49 = 12;
  }
  else
  {
    v48 = v60;
    v49 = 3LL * v61;
  }
  v22 = &v48[v49];
  v23 = v22;
LABEL_31:
  if ( !v19 )
  {
    v21 = v60;
    v20 = v61;
    goto LABEL_33;
  }
  v21 = &v60;
  v25 = 12;
LABEL_34:
  v26 = &v21[v25];
  v27 = v23;
  if ( v26 != v22 )
  {
    while ( 1 )
    {
LABEL_38:
      v28 = *((_DWORD *)v22 + 4);
      if ( v28 )
      {
        v29 = *v22;
        v30 = v22[1] & 0xFFFFFFFFFFFFFFFBLL | (4LL * (v28 <= 0));
        v31 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v31 >= *(_DWORD *)(a3 + 12) )
        {
          v51 = *v22;
          sub_16CD150(a3, a3 + 16, 0, 16);
          v31 = *(unsigned int *)(a3 + 8);
          v29 = v51;
        }
        v32 = (_QWORD *)(*(_QWORD *)a3 + 16 * v31);
        *v32 = v29;
        v32[1] = v30;
        ++*(_DWORD *)(a3 + 8);
      }
      v22 += 3;
      if ( v22 != v27 )
        break;
LABEL_37:
      if ( v22 == v26 )
        goto LABEL_53;
    }
    while ( 1 )
    {
      if ( *v22 == -8 )
      {
        if ( v22[1] != -8 )
          goto LABEL_37;
      }
      else
      {
        if ( *v22 != -16 )
          goto LABEL_37;
        if ( v22[1] != -16 )
        {
          if ( v22 == v26 )
            break;
          goto LABEL_38;
        }
      }
      v22 += 3;
      if ( v27 == v22 )
        goto LABEL_37;
    }
  }
LABEL_53:
  if ( !v54 )
    goto LABEL_69;
  v53 = a3;
  v33 = 0;
  v34 = a1;
  while ( 2 )
  {
    v37 = v34[1];
    v56 = *v34;
    v57 = v37 & 0xFFFFFFFFFFFFFFF8LL;
    v38 = sub_15D0A10((__int64)&v58, &v56, &v55);
    v39 = v55;
    if ( !v38 )
    {
      ++v58;
      v40 = ((unsigned int)v59 >> 1) + 1;
      if ( (v59 & 1) != 0 )
      {
        v35 = 4;
        if ( 4 * v40 >= 0xC )
        {
LABEL_64:
          v41 = 2 * v35;
          goto LABEL_65;
        }
      }
      else
      {
        v35 = v61;
        if ( 4 * v40 >= 3 * v61 )
          goto LABEL_64;
      }
      if ( v35 - (v40 + HIDWORD(v59)) <= v35 >> 3 )
      {
        v41 = v35;
LABEL_65:
        sub_15D0B40((__int64)&v58, v41);
        sub_15D0A10((__int64)&v58, &v56, &v55);
        v39 = v55;
        v40 = ((unsigned int)v59 >> 1) + 1;
      }
      LODWORD(v59) = v59 & 1 | (2 * v40);
      if ( *v39 != -8 || v39[1] != -8 )
        --HIDWORD(v59);
      *v39 = v56;
      v36 = v57;
      *((_DWORD *)v39 + 4) = 0;
      v39[1] = v36;
    }
    *((_DWORD *)v39 + 4) = v33;
    v34 += 2;
    if ( ++v33 != v54 )
      continue;
    break;
  }
  a3 = v53;
LABEL_69:
  v42 = *(__m128i **)a3;
  v43 = *(unsigned int *)(a3 + 8);
  v44 = &v42[v43];
  if ( &v42[v43] != v42 )
  {
    _BitScanReverse64(&v45, (v43 * 16) >> 4);
    sub_15DA890(v42, v42[v43].m128i_i64, 2LL * (int)(63 - (v45 ^ 0x3F)), (__int64)&v58);
    if ( (unsigned __int64)v43 <= 16 )
    {
      sub_15D1080(v42->m128i_i64, v42[v43].m128i_i64, (__int64)&v58);
    }
    else
    {
      v46 = v42 + 16;
      sub_15D1080(v42->m128i_i64, v42[16].m128i_i64, (__int64)&v58);
      if ( v44 != &v42[16] )
      {
        do
        {
          v47 = v46++;
          sub_15D96B0(v47, (__int64)&v58);
        }
        while ( v44 != v46 );
      }
    }
  }
  if ( (v59 & 1) == 0 )
    j___libc_free_0(v60);
}
