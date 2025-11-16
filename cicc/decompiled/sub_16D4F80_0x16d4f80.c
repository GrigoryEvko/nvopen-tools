// Function: sub_16D4F80
// Address: 0x16d4f80
//
void __fastcall sub_16D4F80(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  volatile signed __int32 *v3; // r12
  signed __int32 v4; // eax
  __int64 v5; // r12
  __m128i *v6; // rax
  __int64 v7; // rax
  __int64 (__fastcall ***v8)(__int64); // r13
  __int64 (__fastcall *v9)(__int64); // rdx
  signed __int32 v10; // eax
  __int64 v11; // [rsp+18h] [rbp-98h]
  _QWORD v12[2]; // [rsp+20h] [rbp-90h] BYREF
  int v13; // [rsp+30h] [rbp-80h]
  __int64 v14; // [rsp+38h] [rbp-78h]
  _QWORD v15[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v16; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v17[2]; // [rsp+60h] [rbp-50h] BYREF
  _OWORD v18[4]; // [rsp+70h] [rbp-40h] BYREF

  v1 = *a1;
  if ( *a1 )
  {
    v2 = a1[1];
    if ( !v2 || *(_DWORD *)(v2 + 8) != 1 )
    {
      v5 = *(_QWORD *)(v1 + 32);
      *(_QWORD *)(v1 + 32) = 0;
      if ( v5 )
      {
        v11 = sub_222D560();
        (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*(_QWORD *)v11 + 32LL))(v15, v11, 4);
        v6 = (__m128i *)sub_2241130(v15, 0, 0, "std::future_error: ", 19);
        v17[0] = v18;
        if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
        {
          v18[0] = _mm_loadu_si128(v6 + 1);
        }
        else
        {
          v17[0] = v6->m128i_i64[0];
          *(_QWORD *)&v18[0] = v6[1].m128i_i64[0];
        }
        v17[1] = v6->m128i_i64[1];
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        v6->m128i_i64[1] = 0;
        v6[1].m128i_i8[0] = 0;
        sub_2223530(v12, v17);
        if ( (_OWORD *)v17[0] != v18 )
          j_j___libc_free_0(v17[0], *(_QWORD *)&v18[0] + 1LL);
        if ( (__int64 *)v15[0] != &v16 )
          j_j___libc_free_0(v15[0], v16 + 1);
        v13 = 4;
        v12[0] = off_4A06718;
        v14 = v11;
        sub_2207490(v15);
        v7 = v15[0];
        v15[0] = 0;
        v17[0] = v7;
        sub_22074F0(v17, v5 + 8);
        sub_22074E0(v17);
        sub_22074E0(v15);
        sub_222D210(v12);
        v8 = *(__int64 (__fastcall ****)(__int64))(v1 + 8);
        *(_QWORD *)(v1 + 8) = v5;
        if ( _InterlockedExchange((volatile __int32 *)(v1 + 16), 1) < 0 )
          sub_222D1B0();
        if ( v8 )
        {
          v9 = **v8;
          if ( v9 == sub_16D4120 )
            (*v8)[2]((__int64)v8);
          else
            v9((__int64)v8);
        }
      }
    }
  }
  v3 = (volatile signed __int32 *)a1[1];
  if ( v3 )
  {
    if ( &_pthread_key_create )
    {
      v4 = _InterlockedExchangeAdd(v3 + 2, 0xFFFFFFFF);
    }
    else
    {
      v4 = *((_DWORD *)v3 + 2);
      *((_DWORD *)v3 + 2) = v4 - 1;
    }
    if ( v4 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 16LL))(v3);
      if ( &_pthread_key_create )
      {
        v10 = _InterlockedExchangeAdd(v3 + 3, 0xFFFFFFFF);
      }
      else
      {
        v10 = *((_DWORD *)v3 + 3);
        *((_DWORD *)v3 + 3) = v10 - 1;
      }
      if ( v10 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 24LL))(v3);
    }
  }
}
