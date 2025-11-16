// Function: sub_B07920
// Address: 0xb07920
//
const __m128i *__fastcall sub_B07920(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        char a6,
        const __m128i a7,
        __int64 a8)
{
  __int64 *v10; // r13
  unsigned int v11; // r12d
  __int64 v12; // rbx
  __int64 v13; // r9
  char v14; // r10
  __int64 v15; // rax
  __m128i v16; // xmm1
  __int64 v17; // rbx
  __m128i *v18; // rax
  const __m128i *v19; // r15
  const __m128i *result; // rax
  __int64 v21; // r11
  int v22; // eax
  int v23; // eax
  int v24; // r8d
  unsigned int i; // ebx
  __int64 *v26; // r12
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdi
  int v33; // [rsp+Ch] [rbp-D4h]
  int v34; // [rsp+18h] [rbp-C8h]
  __int64 v35; // [rsp+20h] [rbp-C0h]
  char v36; // [rsp+28h] [rbp-B8h]
  __int64 v38; // [rsp+30h] [rbp-B0h]
  char v39; // [rsp+30h] [rbp-B0h]
  int v40; // [rsp+38h] [rbp-A8h]
  __int64 v41; // [rsp+38h] [rbp-A8h]
  __int64 v42; // [rsp+40h] [rbp-A0h]
  int v44; // [rsp+5Ch] [rbp-84h] BYREF
  __int128 v45; // [rsp+60h] [rbp-80h] BYREF
  __int64 v46; // [rsp+70h] [rbp-70h]
  __int64 v47; // [rsp+80h] [rbp-60h] BYREF
  __int64 v48; // [rsp+88h] [rbp-58h] BYREF
  __int128 v49; // [rsp+90h] [rbp-50h]
  __int64 v50; // [rsp+A0h] [rbp-40h]
  __int64 v51[7]; // [rsp+A8h] [rbp-38h] BYREF

  v10 = a1;
  v11 = a5;
  v12 = a2;
  v13 = a7.m128i_i64[1];
  v14 = a8;
  if ( a5 )
    goto LABEL_2;
  v21 = *a1;
  v47 = a2;
  v48 = a3;
  v46 = a8;
  v50 = a8;
  v51[0] = a4;
  v45 = (__int128)_mm_loadu_si128(&a7);
  v49 = v45;
  v42 = *(_QWORD *)(v21 + 1024);
  v40 = *(_DWORD *)(v21 + 1040);
  if ( v40 )
  {
    if ( (_BYTE)v50 )
    {
      *(_QWORD *)&v45 = *((_QWORD *)&v49 + 1);
      v22 = v49;
    }
    else
    {
      *(_QWORD *)&v45 = 0;
      v22 = 0;
    }
    v35 = v21;
    v36 = a8;
    v38 = a7.m128i_i64[1];
    v44 = v22;
    v23 = sub_AFAA60(&v47, &v48, &v44, (__int64 *)&v45, v51);
    v34 = 1;
    v24 = v40 - 1;
    v41 = v38;
    v33 = v24;
    v39 = v36;
    for ( i = v24 & v23; ; i = v29 & v33 )
    {
      v26 = (__int64 *)(v42 + 8LL * i);
      v27 = *v26;
      if ( *v26 == -4096 )
      {
        v13 = v41;
        v14 = v39;
        v10 = a1;
        v12 = a2;
        v11 = 0;
        goto LABEL_17;
      }
      if ( v27 != -8192 )
      {
        v28 = sub_AF5140(*v26, 0);
        if ( v47 == v28 )
        {
          v30 = sub_AF5140(v27, 1u);
          if ( v48 == v30
            && (_BYTE)v50 == *(_BYTE *)(v27 + 32)
            && (!(_BYTE)v50 || (_DWORD)v49 == *(_DWORD *)(v27 + 16) && *((_QWORD *)&v49 + 1) == *(_QWORD *)(v27 + 24))
            && v51[0] == *(_QWORD *)(v27 + 40) )
          {
            break;
          }
        }
      }
      v29 = i + v34++;
    }
    v31 = v42 + 8LL * i;
    v32 = v27;
    v13 = v41;
    v14 = v39;
    v12 = a2;
    v11 = 0;
    v10 = a1;
    if ( v31 != *(_QWORD *)(v35 + 1024) + 8LL * *(unsigned int *)(v35 + 1040) )
      return (const __m128i *)v32;
  }
LABEL_17:
  result = 0;
  if ( a6 )
  {
LABEL_2:
    v47 = v12;
    if ( !v14 )
      v13 = 0;
    v15 = *v10;
    v16 = _mm_loadu_si128(&a7);
    v48 = a3;
    v17 = v15 + 1016;
    *(_QWORD *)&v49 = v13;
    *((_QWORD *)&v49 + 1) = a4;
    v46 = a8;
    v45 = (__int128)v16;
    v18 = (__m128i *)sub_B97910(48, 4, v11);
    v19 = v18;
    if ( v18 )
      sub_AF2F00(v18, (int)v10, v11, a4, (int)&v47, 4, v45, v46);
    return sub_B07720(v19, v11, v17);
  }
  return result;
}
