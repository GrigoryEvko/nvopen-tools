// Function: sub_2BEDA60
// Address: 0x2beda60
//
__int64 __fastcall sub_2BEDA60(__int64 a1, unsigned __int64 *a2)
{
  char *v2; // r12
  char *v3; // rax
  char *v4; // rdx
  __m128i *v5; // rax
  __int64 v6; // rcx
  __m128i *v7; // r14
  __int64 v8; // r13
  volatile signed __int32 *v9; // rax
  unsigned __int64 v10; // rdi
  volatile signed __int32 *v11; // rax
  unsigned __int64 *v12; // r14
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  volatile signed __int32 *v15; // rbx
  signed __int32 v16; // edx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned int v20; // r12d
  volatile signed __int32 *v21; // r13
  signed __int32 v22; // edx
  signed __int32 v24; // eax
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  signed __int32 v27; // eax
  char *v29; // [rsp+38h] [rbp-258h]
  volatile signed __int32 *v30; // [rsp+48h] [rbp-248h] BYREF
  char *v31; // [rsp+50h] [rbp-240h] BYREF
  size_t v32; // [rsp+58h] [rbp-238h]
  _QWORD v33[2]; // [rsp+60h] [rbp-230h] BYREF
  int v34; // [rsp+70h] [rbp-220h] BYREF
  volatile signed __int32 *v35[2]; // [rsp+78h] [rbp-218h] BYREF
  volatile signed __int32 *v36; // [rsp+88h] [rbp-208h]
  __int64 v37[2]; // [rsp+90h] [rbp-200h] BYREF
  _QWORD v38[2]; // [rsp+A0h] [rbp-1F0h] BYREF
  __m128i *v39; // [rsp+B0h] [rbp-1E0h]
  __int64 v40; // [rsp+B8h] [rbp-1D8h]
  __m128i v41; // [rsp+C0h] [rbp-1D0h] BYREF
  unsigned __int64 v42[2]; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 v43; // [rsp+E0h] [rbp-1B0h]
  __int64 v44; // [rsp+E8h] [rbp-1A8h]
  __int64 *v45; // [rsp+1A0h] [rbp-F0h]
  __int64 v46; // [rsp+1B0h] [rbp-E0h] BYREF
  volatile signed __int32 *v47; // [rsp+1D0h] [rbp-C0h]
  volatile signed __int32 *v48; // [rsp+1D8h] [rbp-B8h]
  __int64 *v49; // [rsp+1E0h] [rbp-B0h]
  __int64 v50; // [rsp+1F0h] [rbp-A0h] BYREF
  unsigned __int64 v51; // [rsp+200h] [rbp-90h]
  unsigned __int64 v52; // [rsp+228h] [rbp-68h]
  __int64 v53; // [rsp+248h] [rbp-48h]

  v29 = 0;
  while ( 1 )
  {
    if ( v29 )
    {
      v2 = v29 + 1;
      v29 = sub_22417D0(&qword_50106C8[8], 44, (unsigned __int64)(v29 + 1));
      v3 = (char *)(v29 - v2);
      if ( (unsigned __int64)v2 > qword_50106C8[9] )
        sub_222CF80(
          "%s: __pos (which is %zu) > this->size() (which is %zu)",
          "basic_string::substr",
          (size_t)v2,
          qword_50106C8[9]);
      v4 = (char *)(qword_50106C8[9] - (_QWORD)v2);
    }
    else
    {
      v2 = 0;
      v3 = sub_22417D0(&qword_50106C8[8], 44, 0);
      v29 = v3;
      v4 = (char *)qword_50106C8[9];
    }
    v31 = (char *)v33;
    if ( v3 > v4 )
      v3 = v4;
    sub_2BDC240((__int64 *)&v31, &v2[qword_50106C8[8]], (__int64)&v2[qword_50106C8[8] + (_QWORD)v3]);
    if ( !v32 )
    {
      v20 = 0;
      goto LABEL_55;
    }
    v37[0] = (__int64)v38;
    sub_2BDC240(v37, ".*", (__int64)"");
    v5 = (__m128i *)sub_2241490((unsigned __int64 *)v37, v31, v32);
    v39 = &v41;
    if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
    {
      v41 = _mm_loadu_si128(v5 + 1);
    }
    else
    {
      v39 = (__m128i *)v5->m128i_i64[0];
      v41.m128i_i64[0] = v5[1].m128i_i64[0];
    }
    v6 = v5->m128i_i64[1];
    v5[1].m128i_i8[0] = 0;
    v40 = v6;
    v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
    v7 = v39;
    v5->m128i_i64[1] = 0;
    v8 = v40;
    sub_220A990(&v30);
    v34 = 16;
    sub_2208E20(v35, &v30);
    if ( !v8 )
      v7 = 0;
    sub_2BED190((__int64)v42, v7, (__int64)v7->m128i_i64 + v8, v35, v34);
    v9 = v47;
    v10 = v51;
    v47 = 0;
    v35[1] = v9;
    v11 = v48;
    v48 = 0;
    v36 = v11;
    if ( v51 )
    {
      v12 = (unsigned __int64 *)v52;
      v13 = v53 + 8;
      if ( v53 + 8 > v52 )
      {
        do
        {
          v14 = *v12++;
          j_j___libc_free_0(v14);
        }
        while ( v13 > (unsigned __int64)v12 );
        v10 = v51;
      }
      j_j___libc_free_0(v10);
    }
    if ( v49 != &v50 )
      j_j___libc_free_0((unsigned __int64)v49);
    v15 = v48;
    if ( v48 )
    {
      if ( &_pthread_key_create )
      {
        v16 = _InterlockedExchangeAdd(v48 + 2, 0xFFFFFFFF);
      }
      else
      {
        v16 = *((_DWORD *)v48 + 2);
        *((_DWORD *)v48 + 2) = v16 - 1;
      }
      if ( v16 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
        if ( &_pthread_key_create )
        {
          v24 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
        }
        else
        {
          v24 = *((_DWORD *)v15 + 3);
          *((_DWORD *)v15 + 3) = v24 - 1;
        }
        if ( v24 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
      }
    }
    if ( v45 != &v46 )
      j_j___libc_free_0((unsigned __int64)v45);
    sub_2209150(&v30);
    if ( v39 != &v41 )
      j_j___libc_free_0((unsigned __int64)v39);
    if ( (_QWORD *)v37[0] != v38 )
      j_j___libc_free_0(v37[0]);
    v42[0] = 0;
    v17 = *a2;
    v18 = a2[1];
    v42[1] = 0;
    v43 = 0;
    v44 = 0;
    v19 = v17 + v18;
    v20 = sub_2BE29D0(v17, v19, v42, (__int64)&v34, 0);
    if ( v42[0] )
    {
      v19 = v43 - v42[0];
      j_j___libc_free_0(v42[0]);
    }
    v21 = v36;
    if ( (_BYTE)v20 )
      break;
    if ( v36 )
    {
      if ( &_pthread_key_create )
      {
        v22 = _InterlockedExchangeAdd(v36 + 2, 0xFFFFFFFF);
      }
      else
      {
        v22 = *((_DWORD *)v36 + 2);
        *((_DWORD *)v36 + 2) = v22 - 1;
      }
      if ( v22 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v21 + 16LL))(v21, v19);
        if ( &_pthread_key_create )
        {
          v25 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF);
        }
        else
        {
          v25 = *((_DWORD *)v21 + 3);
          *((_DWORD *)v21 + 3) = v25 - 1;
        }
        if ( v25 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 24LL))(v21);
      }
    }
    sub_2209150(v35);
    if ( v31 != (char *)v33 )
      j_j___libc_free_0((unsigned __int64)v31);
    if ( v29 == (char *)-1LL )
      return v20;
  }
  if ( v36 )
  {
    if ( &_pthread_key_create )
    {
      v26 = _InterlockedExchangeAdd(v36 + 2, 0xFFFFFFFF);
    }
    else
    {
      v26 = *((_DWORD *)v36 + 2);
      *((_DWORD *)v36 + 2) = v26 - 1;
    }
    if ( v26 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v21 + 16LL))(v21, v19);
      if ( &_pthread_key_create )
      {
        v27 = _InterlockedExchangeAdd(v21 + 3, 0xFFFFFFFF);
      }
      else
      {
        v27 = *((_DWORD *)v21 + 3);
        *((_DWORD *)v21 + 3) = v27 - 1;
      }
      if ( v27 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v21 + 24LL))(v21);
    }
  }
  sub_2209150(v35);
LABEL_55:
  if ( v31 != (char *)v33 )
    j_j___libc_free_0((unsigned __int64)v31);
  return v20;
}
