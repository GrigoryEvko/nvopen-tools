// Function: sub_326BE70
// Address: 0x326be70
//
__int64 __fastcall sub_326BE70(__int64 a1, __int64 a2)
{
  int v2; // r11d
  __int64 v3; // r12
  unsigned __int16 *v4; // rax
  __int64 *v5; // rbx
  unsigned __int64 v6; // r9
  __int64 *v7; // r14
  __int64 *v8; // rdx
  __int64 v9; // r15
  unsigned __int16 v10; // r13
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 v15; // rdx
  const __m128i *v16; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  const __m128i *v19; // r13
  __int64 v20; // r8
  unsigned __int64 v21; // rdx
  __m128i *v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  _BYTE *v26; // r14
  __int64 v27; // r15
  __int64 result; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // edx
  unsigned int v32; // r13d
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r15
  int v36; // ecx
  int v37; // r13d
  _BYTE *v38; // rdx
  __int128 v39; // [rsp-10h] [rbp-E0h]
  __int64 v40; // [rsp+0h] [rbp-D0h]
  int v41; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v42; // [rsp+1Eh] [rbp-B2h]
  __int64 v43; // [rsp+20h] [rbp-B0h]
  int v44; // [rsp+38h] [rbp-98h]
  int v45; // [rsp+38h] [rbp-98h]
  __int64 v46; // [rsp+38h] [rbp-98h]
  int v47; // [rsp+38h] [rbp-98h]
  __int64 v48; // [rsp+40h] [rbp-90h]
  __int64 v49; // [rsp+48h] [rbp-88h]
  int v50; // [rsp+48h] [rbp-88h]
  __int64 v51; // [rsp+48h] [rbp-88h]
  __int64 v52; // [rsp+48h] [rbp-88h]
  __int64 v53; // [rsp+50h] [rbp-80h] BYREF
  int v54; // [rsp+58h] [rbp-78h]
  _BYTE *v55; // [rsp+60h] [rbp-70h] BYREF
  __int64 v56; // [rsp+68h] [rbp-68h]
  _BYTE v57[96]; // [rsp+70h] [rbp-60h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = *(unsigned __int16 **)(a1 + 48);
  v5 = *(__int64 **)(a1 + 40);
  v6 = *v4;
  v48 = *((_QWORD *)v4 + 1);
  v7 = &v5[5 * *(unsigned int *)(a1 + 64)];
  if ( v7 != v5 )
  {
    v8 = *(__int64 **)(a1 + 40);
    v9 = 0;
    v10 = 0;
    v49 = 0;
    while ( 1 )
    {
      v11 = *v8;
      v12 = *(_DWORD *)(*v8 + 24);
      if ( v12 == 51 )
      {
LABEL_8:
        v8 += 5;
        if ( v7 == v8 )
          goto LABEL_9;
      }
      else
      {
        if ( v12 != 159 )
          return 0;
        if ( v9 )
        {
          v13 = *(_QWORD *)(**(_QWORD **)(v11 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v11 + 40) + 8LL);
          if ( *(_WORD *)v13 != v10 )
            return 0;
          if ( v10 )
            goto LABEL_8;
          if ( *(_QWORD *)(v13 + 8) != v49 )
            return 0;
          v8 += 5;
          if ( v7 == v8 )
          {
LABEL_9:
            v43 = v9;
            v55 = v57;
            v56 = 0x300000000LL;
            v42 = v10;
            v41 = v6;
            v14 = v5;
            do
            {
              v15 = *v14;
              if ( *(_DWORD *)(*v14 + 24) == 51 )
              {
                v53 = 0;
                v54 = 0;
                v30 = sub_33F17F0(a2, 51, &v53, v42, v49);
                v32 = v31;
                if ( v53 )
                {
                  v46 = v30;
                  sub_B91220((__int64)&v53, v53);
                  v30 = v46;
                }
                v33 = (unsigned int)v56;
                v34 = v32;
                v35 = *(unsigned int *)(v43 + 64);
                v36 = v56;
                v6 = v35 + (unsigned int)v56;
                v37 = *(_DWORD *)(v43 + 64);
                if ( v6 > HIDWORD(v56) )
                {
                  v40 = v30;
                  v47 = v34;
                  sub_C8D5F0((__int64)&v55, v57, v35 + (unsigned int)v56, 0x10u, v34, v6);
                  v33 = (unsigned int)v56;
                  v30 = v40;
                  LODWORD(v34) = v47;
                  v36 = v56;
                }
                v38 = &v55[16 * v33];
                if ( v35 )
                {
                  do
                  {
                    if ( v38 )
                    {
                      *(_QWORD *)v38 = v30;
                      *((_DWORD *)v38 + 2) = v34;
                    }
                    v38 += 16;
                    --v35;
                  }
                  while ( v35 );
                  v36 = v56;
                }
                LODWORD(v56) = v37 + v36;
              }
              else
              {
                v16 = *(const __m128i **)(v15 + 40);
                v17 = (unsigned int)v56;
                v18 = 40LL * *(unsigned int *)(v15 + 64);
                v19 = (const __m128i *)((char *)v16 + v18);
                v20 = 0xCCCCCCCCCCCCCCCDLL * (v18 >> 3);
                v21 = v20 + (unsigned int)v56;
                if ( v21 > HIDWORD(v56) )
                {
                  v45 = v20;
                  sub_C8D5F0((__int64)&v55, v57, v21, 0x10u, v20, v6);
                  v17 = (unsigned int)v56;
                  LODWORD(v20) = v45;
                }
                v22 = (__m128i *)&v55[16 * v17];
                if ( v16 != v19 )
                {
                  do
                  {
                    if ( v22 )
                      *v22 = _mm_loadu_si128(v16);
                    v16 = (const __m128i *)((char *)v16 + 40);
                    ++v22;
                  }
                  while ( v19 != v16 );
                  LODWORD(v17) = v56;
                }
                LODWORD(v56) = v20 + v17;
              }
              v14 += 5;
            }
            while ( v7 != v14 );
            v3 = a1;
            LODWORD(v6) = v41;
            v2 = a2;
            v23 = v55;
            v24 = (unsigned int)v56;
            goto LABEL_21;
          }
        }
        else
        {
          v29 = *(_QWORD *)(**(_QWORD **)(v11 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v11 + 40) + 8LL);
          v10 = *(_WORD *)v29;
          if ( !*(_WORD *)v29 )
            return 0;
          v49 = *(_QWORD *)(v29 + 8);
          if ( !*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * v10 + 112) )
            return 0;
          v8 += 5;
          v9 = v11;
          if ( v7 == v8 )
            goto LABEL_9;
        }
      }
    }
  }
  v23 = v57;
  v24 = 0;
  v56 = 0x300000000LL;
  v55 = v57;
LABEL_21:
  v25 = *(_QWORD *)(v3 + 80);
  v26 = v23;
  v27 = v24;
  v53 = v25;
  if ( v25 )
  {
    v44 = v6;
    v50 = v2;
    sub_B96E90((__int64)&v53, v25, 1);
    LODWORD(v6) = v44;
    v2 = v50;
  }
  *((_QWORD *)&v39 + 1) = v27;
  *(_QWORD *)&v39 = v26;
  v54 = *(_DWORD *)(v3 + 72);
  result = sub_33FC220(v2, 159, (unsigned int)&v53, v6, v48, v6, v39);
  if ( v53 )
  {
    v51 = result;
    sub_B91220((__int64)&v53, v53);
    result = v51;
  }
  if ( v55 != v57 )
  {
    v52 = result;
    _libc_free((unsigned __int64)v55);
    return v52;
  }
  return result;
}
