// Function: sub_1421630
// Address: 0x1421630
//
void __fastcall sub_1421630(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v6; // r11
  char v8; // r13
  __int64 v11; // rsi
  __int64 *v12; // rax
  char v13; // dl
  char v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  __m128i *v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  bool v20; // zf
  int v21; // eax
  char v22; // r12
  __int64 *v23; // r13
  __int64 v24; // r14
  __int64 *v25; // rax
  __int64 v26; // rsi
  char v27; // dl
  char v28; // dl
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rdx
  __int64 v33; // r10
  _QWORD *v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __m128i *v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // rcx
  __int64 **v41; // rdx
  __int64 *v42; // rdi
  unsigned int v43; // r9d
  __int64 *v44; // rcx
  __int64 *v45; // rdi
  unsigned int v46; // r8d
  __int64 *v47; // rcx
  int v48; // edx
  int v49; // r8d
  char v50; // [rsp+0h] [rbp-380h]
  __int64 v52; // [rsp+10h] [rbp-370h]
  __int64 v53; // [rsp+10h] [rbp-370h]
  __m128i v55; // [rsp+20h] [rbp-360h] BYREF
  __int64 v56; // [rsp+30h] [rbp-350h]
  _BYTE *v57; // [rsp+40h] [rbp-340h] BYREF
  __int64 v58; // [rsp+48h] [rbp-338h]
  _BYTE v59[816]; // [rsp+50h] [rbp-330h] BYREF

  v6 = a3;
  v8 = a5;
  v11 = *a2;
  v57 = v59;
  v58 = 0x2000000000LL;
  v12 = *(__int64 **)(a4 + 8);
  if ( *(__int64 **)(a4 + 16) == v12 )
  {
    v45 = &v12[*(unsigned int *)(a4 + 28)];
    v46 = *(_DWORD *)(a4 + 28);
    if ( v12 != v45 )
    {
      v47 = 0;
      do
      {
        if ( v11 == *v12 )
          goto LABEL_3;
        if ( *v12 == -2 )
          v47 = v12;
        ++v12;
      }
      while ( v45 != v12 );
      if ( v47 )
      {
        *v47 = v11;
        --*(_DWORD *)(a4 + 32);
        ++*(_QWORD *)a4;
        goto LABEL_4;
      }
    }
    if ( v46 < *(_DWORD *)(a4 + 24) )
    {
      *(_DWORD *)(a4 + 28) = v46 + 1;
      *v45 = v11;
      ++*(_QWORD *)a4;
      goto LABEL_4;
    }
  }
  v50 = a6;
  sub_16CCBA0(a4, v11);
  v6 = a3;
  a6 = v50;
  v8 &= v13 ^ 1;
LABEL_3:
  if ( !v8 )
  {
LABEL_4:
    v14 = a6;
    v52 = sub_14214B0(a1, *a2, v6, a6);
    sub_1421110(a1, *a2, v52, v14);
    v15 = a2[3];
    v55.m128i_i64[0] = (__int64)a2;
    v56 = v52;
    v16 = (unsigned int)v58;
    v55.m128i_i64[1] = v15;
    if ( (unsigned int)v58 >= HIDWORD(v58) )
    {
      sub_16CD150(&v57, v59, 0, 24);
      v16 = (unsigned int)v58;
    }
    v17 = (__m128i *)&v57[24 * v16];
    v18 = v56;
    *v17 = _mm_loadu_si128(&v55);
    v19 = v57;
    v17[1].m128i_i64[0] = v18;
    v20 = (_DWORD)v58 == -1;
    v21 = v58 + 1;
    LODWORD(v58) = v58 + 1;
    if ( !v20 )
    {
      v22 = v14;
      do
      {
        while ( 1 )
        {
          v40 = &v19[24 * v21 - 24];
          v41 = (__int64 **)v40[1];
          if ( v41 != *(__int64 ***)(*v40 + 32LL) )
            break;
          LODWORD(v58) = --v21;
          if ( !v21 )
            goto LABEL_22;
        }
        v23 = *v41;
        v24 = v40[2];
        v40[1] = v41 + 1;
        v25 = *(__int64 **)(a4 + 8);
        v26 = *v23;
        if ( *(__int64 **)(a4 + 16) != v25 )
          goto LABEL_9;
        v42 = &v25[*(unsigned int *)(a4 + 28)];
        v43 = *(_DWORD *)(a4 + 28);
        if ( v25 != v42 )
        {
          v44 = 0;
          do
          {
            if ( v26 == *v25 )
            {
              v28 = a5;
              goto LABEL_10;
            }
            if ( *v25 == -2 )
              v44 = v25;
            ++v25;
          }
          while ( v42 != v25 );
          if ( !v44 )
            goto LABEL_37;
          *v44 = v26;
          --*(_DWORD *)(a4 + 32);
          ++*(_QWORD *)a4;
          goto LABEL_27;
        }
LABEL_37:
        if ( v43 < *(_DWORD *)(a4 + 24) )
        {
          *(_DWORD *)(a4 + 28) = v43 + 1;
          *v42 = v26;
          ++*(_QWORD *)a4;
        }
        else
        {
LABEL_9:
          v53 = *v23;
          sub_16CCBA0(a4, v26);
          v26 = v53;
          v28 = a5 & (v27 ^ 1);
LABEL_10:
          if ( v28 )
          {
            v29 = *(unsigned int *)(a1 + 112);
            if ( (_DWORD)v29 )
            {
              v30 = *(_QWORD *)(a1 + 96);
              v31 = (v29 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v32 = (__int64 *)(v30 + 16LL * v31);
              v33 = *v32;
              if ( v26 == *v32 )
              {
LABEL_13:
                if ( v32 != (__int64 *)(v30 + 16 * v29) )
                {
                  v34 = (_QWORD *)v32[1];
                  if ( v34 )
                  {
                    v35 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
                    v24 = v35 - 48;
                    if ( !v35 )
                      v24 = 0;
                  }
                }
              }
              else
              {
                v48 = 1;
                while ( v33 != -8 )
                {
                  v49 = v48 + 1;
                  v31 = (v29 - 1) & (v48 + v31);
                  v32 = (__int64 *)(v30 + 16LL * v31);
                  v33 = *v32;
                  if ( v26 == *v32 )
                    goto LABEL_13;
                  v48 = v49;
                }
              }
            }
            goto LABEL_17;
          }
        }
LABEL_27:
        v24 = sub_14214B0(a1, v26, v24, v22);
LABEL_17:
        sub_1421110(a1, v26, v24, v22);
        v36 = v23[3];
        v55.m128i_i64[0] = (__int64)v23;
        v56 = v24;
        v55.m128i_i64[1] = v36;
        v37 = (unsigned int)v58;
        if ( (unsigned int)v58 >= HIDWORD(v58) )
        {
          sub_16CD150(&v57, v59, 0, 24);
          v37 = (unsigned int)v58;
        }
        v38 = (__m128i *)&v57[24 * v37];
        v39 = v56;
        *v38 = _mm_loadu_si128(&v55);
        v19 = v57;
        v38[1].m128i_i64[0] = v39;
        v21 = v58 + 1;
        LODWORD(v58) = v21;
      }
      while ( v21 );
    }
LABEL_22:
    if ( v19 != v59 )
      goto LABEL_23;
    return;
  }
  v19 = v57;
  if ( v57 != v59 )
LABEL_23:
    _libc_free((unsigned __int64)v19);
}
