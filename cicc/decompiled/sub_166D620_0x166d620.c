// Function: sub_166D620
// Address: 0x166d620
//
__int64 __fastcall sub_166D620(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, size_t a6)
{
  size_t v8; // rax
  _OWORD *v9; // rdx
  size_t v10; // rdx
  __int64 v12; // rax
  _OWORD *v13; // rdi
  _BYTE *v14; // rax
  _QWORD *v15; // rax
  __int64 *v16; // r8
  _QWORD *v17; // r12
  __int64 v18; // rax
  volatile signed __int32 *v19; // rdi
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __m128i *v25; // rax
  __int64 v26; // rcx
  _QWORD *v27; // rsi
  __int64 v28; // rdx
  __m128i *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i *v32; // rdx
  size_t v33; // rcx
  __int64 *v34; // [rsp+0h] [rbp-E0h]
  _BYTE *srca; // [rsp+8h] [rbp-D8h]
  volatile signed __int32 *src; // [rsp+8h] [rbp-D8h]
  _QWORD *dest; // [rsp+10h] [rbp-D0h]
  size_t v38; // [rsp+18h] [rbp-C8h]
  _QWORD v39[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD *v40; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+38h] [rbp-A8h]
  _QWORD v42[2]; // [rsp+40h] [rbp-A0h] BYREF
  char *v43; // [rsp+50h] [rbp-90h] BYREF
  __int64 v44; // [rsp+58h] [rbp-88h]
  char v45; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v46; // [rsp+70h] [rbp-70h] BYREF
  __int64 v47; // [rsp+78h] [rbp-68h]
  __m128i v48; // [rsp+80h] [rbp-60h] BYREF
  _OWORD *v49; // [rsp+90h] [rbp-50h] BYREF
  size_t n; // [rsp+98h] [rbp-48h]
  _OWORD v51[4]; // [rsp+A0h] [rbp-40h] BYREF

  dest = v39;
  LOBYTE(v39[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(v51[0]) = 0;
    v10 = 0;
    v49 = v51;
    goto LABEL_8;
  }
  v46 = (__m128i *)a6;
  v8 = a6;
  v49 = v51;
  if ( a6 > 0xF )
  {
    srca = a5;
    v12 = sub_22409D0(&v49, &v46, 0);
    a5 = srca;
    v49 = (_OWORD *)v12;
    v13 = (_OWORD *)v12;
    *(_QWORD *)&v51[0] = v46;
LABEL_16:
    memcpy(v13, a5, a6);
    v8 = (size_t)v46;
    v9 = v49;
    goto LABEL_5;
  }
  if ( a6 != 1 )
  {
    if ( !a6 )
    {
      v9 = v51;
      goto LABEL_5;
    }
    v13 = v51;
    goto LABEL_16;
  }
  LOBYTE(v51[0]) = *a5;
  v9 = v51;
LABEL_5:
  n = v8;
  *((_BYTE *)v9 + v8) = 0;
  if ( v49 != v51 )
  {
    dest = v49;
    v38 = n;
    v39[0] = *(_QWORD *)&v51[0];
    v49 = v51;
    v14 = v51;
    goto LABEL_9;
  }
  v10 = n;
  if ( n )
  {
    if ( n == 1 )
      LOBYTE(v39[0]) = v51[0];
    else
      memcpy(v39, v51, n);
    v10 = n;
  }
LABEL_8:
  v38 = v10;
  *((_BYTE *)v39 + v10) = 0;
  v14 = v49;
LABEL_9:
  n = 0;
  *v14 = 0;
  if ( v49 != v51 )
    j_j___libc_free_0(v49, *(_QWORD *)&v51[0] + 1LL);
  if ( v38 )
  {
    src = *(volatile signed __int32 **)(a1 + 160);
    v15 = (_QWORD *)sub_22077B0(32);
    v16 = (__int64 *)src;
    v17 = v15;
    if ( v15 )
    {
      v15[1] = 0x100000001LL;
      v34 = (__int64 *)src;
      *v15 = &unk_49EE468;
      src = (volatile signed __int32 *)(v15 + 2);
      sub_16C9340(v15 + 2, dest, v38, 0);
      v16 = v34;
      v18 = (__int64)(v17 + 2);
    }
    else
    {
      v18 = 16;
    }
    v19 = (volatile signed __int32 *)v16[1];
    *v16 = v18;
    v16[1] = (__int64)v17;
    if ( v19 )
    {
      if ( &_pthread_key_create )
      {
        v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
      }
      else
      {
        v20 = *((_DWORD *)v19 + 2);
        *((_DWORD *)v19 + 2) = v20 - 1;
      }
      if ( v20 == 1 )
      {
        v34 = v16;
        src = v19;
        (*(void (**)(void))(*(_QWORD *)v19 + 16LL))();
        v16 = v34;
        if ( &_pthread_key_create )
        {
          v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
        }
        else
        {
          v21 = *((_DWORD *)v19 + 3);
          *((_DWORD *)v19 + 3) = v21 - 1;
        }
        if ( v21 == 1 )
        {
          src = (volatile signed __int32 *)v34;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
          v16 = v34;
        }
      }
    }
    LOBYTE(v42[0]) = 0;
    v40 = v42;
    v41 = 0;
    if ( !(unsigned __int8)sub_16C9430(*v16, &v40) )
    {
      v45 = 0;
      v43 = &v45;
      v44 = 0;
      sub_2240E30(&v43, v38 + 28);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v44) > 0x1B )
      {
        sub_2241490(&v43, "Invalid regular expression '", 28, v22);
        sub_2241490(&v43, dest, v38, v23);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v44) > 0x13 )
        {
          v25 = (__m128i *)sub_2241490(&v43, "' in -pass-remarks: ", 20, v24);
          v46 = &v48;
          if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
          {
            v48 = _mm_loadu_si128(v25 + 1);
          }
          else
          {
            v46 = (__m128i *)v25->m128i_i64[0];
            v48.m128i_i64[0] = v25[1].m128i_i64[0];
          }
          v47 = v25->m128i_i64[1];
          v26 = v47;
          v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
          v27 = v40;
          v25->m128i_i64[1] = 0;
          v28 = v41;
          v25[1].m128i_i8[0] = 0;
          v29 = (__m128i *)sub_2241490(&v46, v27, v28, v26);
          v49 = v51;
          v32 = v29 + 1;
          if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
          {
            v51[0] = _mm_loadu_si128(v29 + 1);
          }
          else
          {
            v49 = (_OWORD *)v29->m128i_i64[0];
            *(_QWORD *)&v51[0] = v29[1].m128i_i64[0];
          }
          n = v29->m128i_u64[1];
          v33 = n;
          v29->m128i_i64[0] = (__int64)v32;
          v29->m128i_i64[1] = 0;
          v29[1].m128i_i8[0] = 0;
          sub_16BD160(&v49, 0, v32, v33, v30, v31, v34, src, dest, v38);
        }
      }
      sub_4262D8((__int64)"basic_string::append");
    }
    if ( v40 != v42 )
      j_j___libc_free_0(v40, v42[0] + 1LL);
  }
  *(_DWORD *)(a1 + 16) = a2;
  if ( dest != v39 )
    j_j___libc_free_0(dest, v39[0] + 1LL);
  return 0;
}
