// Function: sub_21C01C0
// Address: 0x21c01c0
//
__int64 __fastcall sub_21C01C0(
        __int64 a1,
        __int64 a2,
        int a3,
        char a4,
        const __m128i *a5,
        __int64 *a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  __int64 v10; // r11
  _DWORD *v11; // r10
  char v12; // r14
  unsigned __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned __int64 v21; // r14
  __int64 v22; // rax
  const __m128i **v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  const __m128i *v27; // r12
  __int64 v28; // rax
  const __m128i *v29; // rax
  unsigned int v30; // edx
  char v32; // al
  char v33; // cl
  char v34; // si
  char v35; // dl
  int v36; // eax
  char v37; // al
  char v38; // al
  char v39; // cl
  char v40; // si
  char v41; // dl
  char v42; // al
  char v43; // cl
  char v44; // si
  char v45; // dl
  char v46; // al
  char v47; // cl
  char v48; // si
  char v49; // dl
  char v50; // al
  __int64 v51; // [rsp+8h] [rbp-B8h]
  __int64 v52; // [rsp+8h] [rbp-B8h]
  __int64 v53; // [rsp+10h] [rbp-B0h]
  __int64 v54; // [rsp+10h] [rbp-B0h]
  __int64 v57; // [rsp+30h] [rbp-90h]
  __int64 v58; // [rsp+30h] [rbp-90h]
  __int64 v59; // [rsp+30h] [rbp-90h]
  const __m128i *v60; // [rsp+30h] [rbp-90h]
  const __m128i *v61; // [rsp+30h] [rbp-90h]
  _DWORD *v64; // [rsp+40h] [rbp-80h]
  unsigned __int64 v65; // [rsp+48h] [rbp-78h]
  _DWORD v66[16]; // [rsp+80h] [rbp-40h] BYREF

  v11 = v66;
  v12 = 1;
  v13 = a3;
  v14 = (__int64)a5;
  v66[0] = 0;
  if ( a3 <= 0 )
  {
    if ( *(unsigned int *)(a2 + 8) > (unsigned __int64)a3 )
      goto LABEL_11;
LABEL_20:
    *(_BYTE *)(a1 + 4) = 0;
    goto LABEL_21;
  }
  v65 = a3;
  v15 = 0;
  v16 = v57;
  do
  {
    while ( 1 )
    {
      a5 = (const __m128i *)(*(_QWORD *)a2 + 16 * v15);
      v24 = a5->m128i_i64[0];
      *((_BYTE *)v66 + v15) = (unsigned __int16)(*(_WORD *)(a5->m128i_i64[0] + 24) - 32) <= 1u
                           || (unsigned __int16)(*(_WORD *)(a5->m128i_i64[0] + 24) - 10) <= 1u;
      if ( !*((_BYTE *)v66 + v15) )
        break;
      v17 = a5->m128i_i64[1];
      if ( (unsigned __int8)(a4 - 9) <= 1u )
      {
        v28 = *(_QWORD *)(v24 + 40);
        v59 = v10;
        LOBYTE(v16) = *(_BYTE *)v28;
        v29 = (const __m128i *)sub_1D360F0(
                                 a6,
                                 *(_QWORD *)(v24 + 88),
                                 a10,
                                 (unsigned int)v16,
                                 *(const void ***)(v28 + 8),
                                 1,
                                 *(double *)a7.m128i_i64,
                                 a8,
                                 a9);
        v10 = v59;
        a5 = v29;
        v21 = v30 | v17 & 0xFFFFFFFF00000000LL;
        v22 = *(unsigned int *)(v14 + 8);
        if ( (unsigned int)v22 >= *(_DWORD *)(v14 + 12) )
        {
LABEL_18:
          v51 = v10;
          v53 = v16;
          v60 = a5;
          sub_16CD150(v14, (const void *)(v14 + 16), 0, 16, (int)a5, (int)a6);
          v22 = *(unsigned int *)(v14 + 8);
          v10 = v51;
          v16 = v53;
          a5 = v60;
        }
      }
      else
      {
        v18 = *(_QWORD *)(v24 + 40);
        v58 = v16;
        LOBYTE(v10) = *(_BYTE *)v18;
        v19 = sub_1D37E40(
                (__int64)a6,
                *(_QWORD *)(v24 + 88),
                a10,
                (unsigned int)v10,
                *(const void ***)(v18 + 8),
                1,
                a7,
                a8,
                a9,
                0);
        v16 = v58;
        a5 = (const __m128i *)v19;
        v21 = v20 | v17 & 0xFFFFFFFF00000000LL;
        v22 = *(unsigned int *)(v14 + 8);
        if ( (unsigned int)v22 >= *(_DWORD *)(v14 + 12) )
          goto LABEL_18;
      }
      v23 = (const __m128i **)(*(_QWORD *)v14 + 16 * v22);
      ++v15;
      v23[1] = (const __m128i *)v21;
      v12 = 0;
      *v23 = a5;
      ++*(_DWORD *)(v14 + 8);
      if ( v65 == v15 )
        goto LABEL_10;
    }
    v25 = *(unsigned int *)(v14 + 8);
    if ( (unsigned int)v25 >= *(_DWORD *)(v14 + 12) )
    {
      v52 = v10;
      v54 = v16;
      v61 = a5;
      sub_16CD150(v14, (const void *)(v14 + 16), 0, 16, (int)a5, (int)a6);
      v25 = *(unsigned int *)(v14 + 8);
      v10 = v52;
      v16 = v54;
      a5 = v61;
    }
    a7 = _mm_loadu_si128(a5);
    ++v15;
    *(__m128i *)(*(_QWORD *)v14 + 16 * v25) = a7;
    ++*(_DWORD *)(v14 + 8);
  }
  while ( v65 != v15 );
LABEL_10:
  v11 = v66;
  v13 = v65;
  if ( *(unsigned int *)(a2 + 8) > v65 )
  {
LABEL_11:
    v26 = *(unsigned int *)(v14 + 8);
    do
    {
      v27 = (const __m128i *)(*(_QWORD *)a2 + 16 * v13);
      if ( *(_DWORD *)(v14 + 12) <= (unsigned int)v26 )
      {
        v64 = v11;
        sub_16CD150(v14, (const void *)(v14 + 16), 0, 16, (int)a5, (int)a6);
        v26 = *(unsigned int *)(v14 + 8);
        v11 = v64;
      }
      ++v13;
      *(__m128i *)(*(_QWORD *)v14 + 16 * v26) = _mm_loadu_si128(v27);
      v26 = (unsigned int)(*(_DWORD *)(v14 + 8) + 1);
      *(_DWORD *)(v14 + 8) = v26;
    }
    while ( v13 < *(unsigned int *)(a2 + 8) );
  }
  if ( v12 )
    goto LABEL_20;
  switch ( a4 )
  {
    case 3:
      v46 = *(_BYTE *)v11;
      v47 = *((_BYTE *)v11 + 1);
      if ( a3 == 2 )
      {
        if ( v46 )
          v36 = (v47 == 0) + 4145;
        else
          v36 = (v47 == 0) + 4147;
      }
      else
      {
        v48 = *((_BYTE *)v11 + 2);
        v49 = *((_BYTE *)v11 + 3);
        if ( v46 )
        {
          if ( v47 )
          {
            if ( v48 )
              v36 = (v49 == 0) + 4203;
            else
              v36 = (v49 == 0) + 4205;
          }
          else if ( v48 )
          {
            v36 = (v49 == 0) + 4207;
          }
          else
          {
            v36 = (v49 == 0) + 4209;
          }
        }
        else if ( v47 )
        {
          if ( v48 )
            v36 = (v49 == 0) + 4211;
          else
            v36 = (v49 == 0) + 4213;
        }
        else if ( v48 )
        {
          v36 = (v49 == 0) + 4215;
        }
        else
        {
          v36 = (v49 == 0) + 4217;
        }
      }
      break;
    case 4:
      v42 = *(_BYTE *)v11;
      v43 = *((_BYTE *)v11 + 1);
      if ( a3 == 2 )
      {
        if ( v42 )
          v36 = (v43 == 0) + 4133;
        else
          v36 = (v43 == 0) + 4135;
      }
      else
      {
        v44 = *((_BYTE *)v11 + 2);
        v45 = *((_BYTE *)v11 + 3);
        if ( v42 )
        {
          if ( v43 )
          {
            if ( v44 )
              v36 = (v45 == 0) + 4171;
            else
              v36 = (v45 == 0) + 4173;
          }
          else if ( v44 )
          {
            v36 = (v45 == 0) + 4175;
          }
          else
          {
            v36 = (v45 == 0) + 4177;
          }
        }
        else if ( v43 )
        {
          if ( v44 )
            v36 = (v45 == 0) + 4179;
          else
            v36 = (v45 == 0) + 4181;
        }
        else if ( v44 )
        {
          v36 = (v45 == 0) + 4183;
        }
        else
        {
          v36 = (v45 == 0) + 4185;
        }
      }
      break;
    case 5:
      v38 = *(_BYTE *)v11;
      v39 = *((_BYTE *)v11 + 1);
      if ( a3 == 2 )
      {
        if ( v38 )
          v36 = (v39 == 0) + 4137;
        else
          v36 = (v39 == 0) + 4139;
      }
      else
      {
        v40 = *((_BYTE *)v11 + 2);
        v41 = *((_BYTE *)v11 + 3);
        if ( v38 )
        {
          if ( v39 )
          {
            if ( v40 )
              v36 = (v41 == 0) + 4187;
            else
              v36 = (v41 == 0) + 4189;
          }
          else if ( v40 )
          {
            v36 = (v41 == 0) + 4191;
          }
          else
          {
            v36 = (v41 == 0) + 4193;
          }
        }
        else if ( v39 )
        {
          if ( v40 )
            v36 = (v41 == 0) + 4195;
          else
            v36 = (v41 == 0) + 4197;
        }
        else if ( v40 )
        {
          v36 = (v41 == 0) + 4199;
        }
        else
        {
          v36 = (v41 == 0) + 4201;
        }
      }
      break;
    case 6:
      if ( a3 == 4 )
        goto LABEL_20;
      v37 = *((_BYTE *)v11 + 1);
      if ( *(_BYTE *)v11 )
        v36 = (v37 == 0) + 4141;
      else
        v36 = (v37 == 0) + 4143;
      break;
    case 9:
      v32 = *(_BYTE *)v11;
      v33 = *((_BYTE *)v11 + 1);
      if ( a3 == 2 )
      {
        if ( v32 )
          v36 = (v33 == 0) + 4125;
        else
          v36 = (v33 == 0) + 4127;
      }
      else
      {
        v34 = *((_BYTE *)v11 + 2);
        v35 = *((_BYTE *)v11 + 3);
        if ( v32 )
        {
          if ( v33 )
          {
            if ( v34 )
              v36 = (v35 == 0) + 4155;
            else
              v36 = (v35 == 0) + 4157;
          }
          else if ( v34 )
          {
            v36 = (v35 == 0) + 4159;
          }
          else
          {
            v36 = (v35 == 0) + 4161;
          }
        }
        else if ( v33 )
        {
          if ( v34 )
            v36 = (v35 == 0) + 4163;
          else
            v36 = (v35 == 0) + 4165;
        }
        else if ( v34 )
        {
          v36 = (v35 == 0) + 4167;
        }
        else
        {
          v36 = (v35 == 0) + 4169;
        }
      }
      break;
    case 10:
      if ( a3 == 4 )
        goto LABEL_20;
      v50 = *((_BYTE *)v11 + 1);
      if ( *(_BYTE *)v11 )
        v36 = (v50 == 0) + 4129;
      else
        v36 = (v50 == 0) + 4131;
      break;
    default:
      goto LABEL_20;
  }
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = v36;
LABEL_21:
  if ( v11 != v66 )
    _libc_free((unsigned __int64)v11);
  return a1;
}
