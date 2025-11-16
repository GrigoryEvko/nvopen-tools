// Function: sub_1F6FDD0
// Address: 0x1f6fdd0
//
__int64 *__fastcall sub_1F6FDD0(__int64 **a1, __int64 **a2, unsigned int a3, __m128 a4, double a5, __m128i a6)
{
  __int64 v6; // r12
  __int64 *v8; // r10
  __int64 v9; // r13
  __int64 v10; // rsi
  unsigned __int64 v11; // r9
  __int64 *v12; // r10
  unsigned __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // r8
  _QWORD *v18; // rdi
  _QWORD *v19; // rdx
  _QWORD *v20; // rsi
  unsigned __int64 v21; // r15
  __int64 *v22; // rdx
  __int64 v23; // r12
  _QWORD *v24; // r14
  const __m128i *v25; // r12
  __int64 v26; // rax
  __int64 *v27; // r12
  _QWORD *v29; // rdx
  __int128 v30; // [rsp-10h] [rbp-170h]
  __int64 v32; // [rsp+18h] [rbp-148h]
  __int64 v33; // [rsp+20h] [rbp-140h] BYREF
  int v34; // [rsp+28h] [rbp-138h]
  __int64 v35; // [rsp+30h] [rbp-130h] BYREF
  _BYTE *v36; // [rsp+38h] [rbp-128h]
  _BYTE *v37; // [rsp+40h] [rbp-120h]
  __int64 v38; // [rsp+48h] [rbp-118h]
  int v39; // [rsp+50h] [rbp-110h]
  _BYTE v40[72]; // [rsp+58h] [rbp-108h] BYREF
  _BYTE *v41; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-B8h]
  _BYTE v43[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v6 = a3;
  v41 = v43;
  v8 = *a2;
  v42 = 0x800000000LL;
  v36 = v40;
  v37 = v40;
  v35 = 0;
  v38 = 8;
  v39 = 0;
  v9 = *v8;
  v10 = *(_QWORD *)(*v8 + 72);
  v33 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v33, v10, 2);
  v34 = *(_DWORD *)(v9 + 64);
  if ( (_DWORD)v6 )
  {
    v11 = (unsigned __int64)v37;
    v12 = *a2;
    v13 = 0;
    v14 = v36;
    v15 = (unsigned int)(v6 - 1);
    v16 = 16 * v6;
    while ( 1 )
    {
      v17 = v12[v13 / 8];
      if ( v14 != (_QWORD *)v11 )
        goto LABEL_5;
      v18 = &v14[HIDWORD(v38)];
      if ( v18 == v14 )
        goto LABEL_46;
      v19 = v14;
      v20 = 0;
      do
      {
        if ( v17 == *v19 )
          goto LABEL_6;
        if ( *v19 == -2 )
          v20 = v19;
        ++v19;
      }
      while ( v18 != v19 );
      if ( !v20 )
      {
LABEL_46:
        if ( HIDWORD(v38) < (unsigned int)v38 )
        {
          ++HIDWORD(v38);
          *v18 = v17;
          v14 = v36;
          ++v35;
          v11 = (unsigned __int64)v37;
          v12 = *a2;
          goto LABEL_6;
        }
LABEL_5:
        sub_16CCBA0((__int64)&v35, v12[v13 / 8]);
        v11 = (unsigned __int64)v37;
        v14 = v36;
        v12 = *a2;
LABEL_6:
        v13 += 16LL;
        if ( v16 == v13 )
          goto LABEL_16;
      }
      else
      {
        v13 += 16LL;
        *v20 = v17;
        v11 = (unsigned __int64)v37;
        --v39;
        v14 = v36;
        ++v35;
        v12 = *a2;
        if ( v16 == v13 )
        {
LABEL_16:
          v21 = 0;
          v32 = 16 * v15;
          while ( 1 )
          {
            v22 = *(__int64 **)(v12[v21 / 8] + 32);
            v23 = *v22;
            if ( v14 == (_QWORD *)v11 )
            {
              v24 = &v14[HIDWORD(v38)];
              if ( v24 == v14 )
              {
                v29 = v14;
              }
              else
              {
                do
                {
                  if ( v23 == *v14 )
                    break;
                  ++v14;
                }
                while ( v24 != v14 );
                v29 = v24;
              }
            }
            else
            {
              v24 = (_QWORD *)(v11 + 8LL * (unsigned int)v38);
              v14 = sub_16CC9F0((__int64)&v35, *v22);
              if ( v23 == *v14 )
              {
                if ( v37 == v36 )
                  v29 = &v37[8 * HIDWORD(v38)];
                else
                  v29 = &v37[8 * (unsigned int)v38];
              }
              else
              {
                if ( v37 != v36 )
                {
                  v14 = &v37[8 * (unsigned int)v38];
                  goto LABEL_23;
                }
                v14 = &v37[8 * HIDWORD(v38)];
                v29 = v14;
              }
            }
            while ( v29 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
              ++v14;
LABEL_23:
            if ( v24 == v14 )
            {
              v25 = *(const __m128i **)((*a2)[v21 / 8] + 32);
              v26 = (unsigned int)v42;
              if ( (unsigned int)v42 >= HIDWORD(v42) )
              {
                sub_16CD150((__int64)&v41, v43, 0, 16, v17, v11);
                v26 = (unsigned int)v42;
              }
              a4 = (__m128)_mm_loadu_si128(v25);
              *(__m128 *)&v41[16 * v26] = a4;
              LODWORD(v42) = v42 + 1;
              if ( v21 == v32 )
                goto LABEL_27;
            }
            else if ( v21 == v32 )
            {
              goto LABEL_27;
            }
            v11 = (unsigned __int64)v37;
            v14 = v36;
            v21 += 16LL;
            v12 = *a2;
          }
        }
      }
    }
  }
LABEL_27:
  *((_QWORD *)&v30 + 1) = (unsigned int)v42;
  *(_QWORD *)&v30 = v41;
  v27 = sub_1D359D0(*a1, 2, (__int64)&v33, 1, 0, 0, *(double *)a4.m128_u64, a5, a6, v30);
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  if ( v37 != v36 )
    _libc_free((unsigned __int64)v37);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  return v27;
}
