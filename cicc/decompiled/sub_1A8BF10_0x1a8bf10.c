// Function: sub_1A8BF10
// Address: 0x1a8bf10
//
void __fastcall sub_1A8BF10(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        _QWORD *a14)
{
  __int64 v14; // rdx
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // r13
  _QWORD *v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // rbx
  _BYTE *v25; // r12
  __int64 v26; // r13
  __int64 v27; // rax
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned int v32; // esi
  __int64 v33; // rcx
  _QWORD *v34; // r13
  int v35; // edx
  __int64 v36; // rax
  unsigned __int64 *v37; // r15
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned int v40; // eax
  int v41; // r10d
  int v42; // eax
  int v43; // edi
  int v44; // edi
  int v45; // edx
  unsigned int v46; // eax
  __int64 v47; // rsi
  _QWORD *v48; // rdx
  int v49; // esi
  _QWORD *v50; // rdx
  unsigned int v51; // eax
  __int64 v52; // rdi
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 v54; // [rsp+18h] [rbp-B8h]
  _QWORD v55[2]; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v56; // [rsp+38h] [rbp-98h]
  __int64 v57; // [rsp+40h] [rbp-90h]
  _BYTE *v58; // [rsp+50h] [rbp-80h] BYREF
  __int64 v59; // [rsp+58h] [rbp-78h]
  _BYTE v60[112]; // [rsp+60h] [rbp-70h] BYREF

  v14 = *(_QWORD *)(a1 + 112);
  v58 = v60;
  v59 = 0x800000000LL;
  v53 = *(_QWORD *)(v14 + 40);
  if ( v53 == *(_QWORD *)(v14 + 32) )
    return;
  v54 = *(_QWORD *)(v14 + 32);
  do
  {
    v16 = *(_QWORD *)(*(_QWORD *)v54 + 48LL);
    v17 = *(_QWORD *)v54 + 40LL;
    if ( v17 != v16 )
    {
      while ( 1 )
      {
        v18 = v16 - 24;
        v19 = *(_QWORD **)(a1 + 16);
        if ( !v16 )
          v18 = 0;
        v20 = *(_QWORD **)(a1 + 8);
        if ( v19 == v20 )
          break;
        v21 = &v19[*(unsigned int *)(a1 + 24)];
        v20 = sub_16CC9F0(a1, v18);
        if ( v18 == *v20 )
        {
          v30 = *(_QWORD *)(a1 + 16);
          if ( v30 == *(_QWORD *)(a1 + 8) )
            v31 = *(unsigned int *)(a1 + 28);
          else
            v31 = *(unsigned int *)(a1 + 24);
          v48 = (_QWORD *)(v30 + 8 * v31);
          goto LABEL_30;
        }
        v22 = *(_QWORD *)(a1 + 16);
        if ( v22 == *(_QWORD *)(a1 + 8) )
        {
          v20 = (_QWORD *)(v22 + 8LL * *(unsigned int *)(a1 + 28));
          v48 = v20;
          goto LABEL_30;
        }
        v20 = (_QWORD *)(v22 + 8LL * *(unsigned int *)(a1 + 24));
LABEL_12:
        if ( v20 == v21 )
        {
          if ( *(_DWORD *)(a1 + 224) )
          {
            v56 = v18;
            v55[0] = 2;
            v55[1] = 0;
            if ( v18 != 0 && v18 != -8 && v18 != -16 )
              sub_164C220((__int64)v55);
            v32 = *(_DWORD *)(a1 + 232);
            v57 = a1 + 208;
            if ( !v32 )
            {
              ++*(_QWORD *)(a1 + 208);
              goto LABEL_42;
            }
            v33 = v56;
            v39 = *(_QWORD *)(a1 + 216);
            v40 = (v32 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
            v34 = (_QWORD *)(v39 + ((unsigned __int64)v40 << 6));
            a13 = v34[3];
            if ( v56 == a13 )
              goto LABEL_55;
            v41 = 1;
            a14 = 0;
            while ( a13 != -8 )
            {
              if ( a13 == -16 && !a14 )
                a14 = v34;
              v40 = (v32 - 1) & (v41 + v40);
              v34 = (_QWORD *)(v39 + ((unsigned __int64)v40 << 6));
              a13 = v34[3];
              if ( v56 == a13 )
                goto LABEL_55;
              ++v41;
            }
            v42 = *(_DWORD *)(a1 + 224);
            if ( a14 )
              v34 = a14;
            ++*(_QWORD *)(a1 + 208);
            v35 = v42 + 1;
            if ( 4 * (v42 + 1) >= 3 * v32 )
            {
LABEL_42:
              sub_12E48B0(a1 + 208, 2 * v32);
              LODWORD(a13) = *(_DWORD *)(a1 + 232);
              if ( (_DWORD)a13 )
              {
                v33 = v56;
                LODWORD(a13) = a13 - 1;
                v49 = 1;
                a14 = *(_QWORD **)(a1 + 216);
                v50 = 0;
                v51 = a13 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v34 = &a14[8 * (unsigned __int64)v51];
                v52 = v34[3];
                if ( v52 != v56 )
                {
                  while ( v52 != -8 )
                  {
                    if ( !v50 && v52 == -16 )
                      v50 = v34;
                    v51 = a13 & (v49 + v51);
                    v34 = &a14[8 * (unsigned __int64)v51];
                    v52 = v34[3];
                    if ( v56 == v52 )
                      goto LABEL_44;
                    ++v49;
                  }
                  if ( v50 )
                    v34 = v50;
                }
                goto LABEL_44;
              }
            }
            else
            {
              if ( v32 - *(_DWORD *)(a1 + 228) - v35 > v32 >> 3 )
                goto LABEL_45;
              sub_12E48B0(a1 + 208, v32);
              v43 = *(_DWORD *)(a1 + 232);
              if ( v43 )
              {
                v33 = v56;
                v44 = v43 - 1;
                a13 = *(_QWORD *)(a1 + 216);
                a14 = 0;
                v45 = 1;
                v46 = v44 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v34 = (_QWORD *)(a13 + ((unsigned __int64)v46 << 6));
                v47 = v34[3];
                if ( v47 != v56 )
                {
                  while ( v47 != -8 )
                  {
                    if ( !a14 && v47 == -16 )
                      a14 = v34;
                    v46 = v44 & (v45 + v46);
                    v34 = (_QWORD *)(a13 + ((unsigned __int64)v46 << 6));
                    v47 = v34[3];
                    if ( v56 == v47 )
                      goto LABEL_44;
                    ++v45;
                  }
                  if ( a14 )
                    v34 = a14;
                }
LABEL_44:
                v35 = *(_DWORD *)(a1 + 224) + 1;
LABEL_45:
                *(_DWORD *)(a1 + 224) = v35;
                if ( v34[3] == -8 )
                {
                  v37 = v34 + 1;
                  if ( v33 != -8 )
                    goto LABEL_50;
                }
                else
                {
                  --*(_DWORD *)(a1 + 228);
                  v36 = v34[3];
                  if ( v36 != v33 )
                  {
                    v37 = v34 + 1;
                    if ( v36 != -8 && v36 != 0 && v36 != -16 )
                    {
                      sub_1649B30(v34 + 1);
                      v33 = v56;
                    }
LABEL_50:
                    v34[3] = v33;
                    if ( v33 != -8 && v33 != 0 && v33 != -16 )
                      sub_1649AC0(v37, v55[0] & 0xFFFFFFFFFFFFFFF8LL);
                    v33 = v56;
                  }
                }
                v38 = v57;
                v34[5] = 6;
                v34[6] = 0;
                v34[4] = v38;
                v34[7] = 0;
LABEL_55:
                if ( v33 != 0 && v33 != -8 && v33 != -16 )
                  sub_1649B30(v55);
                v18 = v34[7];
                v23 = (unsigned int)v59;
                if ( (unsigned int)v59 < HIDWORD(v59) )
                  goto LABEL_15;
LABEL_59:
                sub_16CD150((__int64)&v58, v60, 0, 8, a13, (int)a14);
                v23 = (unsigned int)v59;
                goto LABEL_15;
              }
            }
            v33 = v56;
            v34 = 0;
            goto LABEL_44;
          }
          v23 = (unsigned int)v59;
          if ( (unsigned int)v59 >= HIDWORD(v59) )
            goto LABEL_59;
LABEL_15:
          *(_QWORD *)&v58[8 * v23] = v18;
          LODWORD(v59) = v59 + 1;
          v16 = *(_QWORD *)(v16 + 8);
          if ( v17 == v16 )
            goto LABEL_16;
        }
        else
        {
          v16 = *(_QWORD *)(v16 + 8);
          if ( v17 == v16 )
            goto LABEL_16;
        }
      }
      v21 = &v20[*(unsigned int *)(a1 + 28)];
      if ( v20 == v21 )
      {
        v48 = *(_QWORD **)(a1 + 8);
      }
      else
      {
        do
        {
          if ( v18 == *v20 )
            break;
          ++v20;
        }
        while ( v21 != v20 );
        v48 = v21;
      }
LABEL_30:
      while ( v48 != v20 )
      {
        if ( *v20 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v20;
      }
      goto LABEL_12;
    }
LABEL_16:
    v54 += 8;
  }
  while ( v53 != v54 );
  v24 = v58;
  v25 = &v58[8 * (unsigned int)v59];
  if ( v58 != v25 )
  {
    do
    {
      v26 = *((_QWORD *)v25 - 1);
      if ( *(_QWORD *)(v26 + 8) )
      {
        v27 = sub_1599EF0(*(__int64 ***)v26);
        sub_164D160(v26, v27, a2, a3, a4, a5, v28, v29, a8, a9);
      }
      v25 -= 8;
      sub_15F20C0((_QWORD *)v26);
    }
    while ( v24 != v25 );
    v25 = v58;
  }
  if ( v25 != v60 )
    _libc_free((unsigned __int64)v25);
}
