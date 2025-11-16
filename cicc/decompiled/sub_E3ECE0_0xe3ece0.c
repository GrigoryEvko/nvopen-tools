// Function: sub_E3ECE0
// Address: 0xe3ece0
//
void __fastcall sub_E3ECE0(__int64 *a1, __int64 *a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // rdx
  volatile signed __int32 *v8; // r15
  signed __int32 v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  signed __int32 v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD v20[2]; // [rsp-1A8h] [rbp-1A8h] BYREF
  _QWORD v21[2]; // [rsp-198h] [rbp-198h] BYREF
  __m128i v22; // [rsp-188h] [rbp-188h] BYREF
  char v23; // [rsp-168h] [rbp-168h]
  char v24; // [rsp-167h] [rbp-167h]
  __m128i v25; // [rsp-158h] [rbp-158h] BYREF
  __int16 v26; // [rsp-138h] [rbp-138h]
  __m128i v27[3]; // [rsp-128h] [rbp-128h] BYREF
  __m128i v28; // [rsp-F8h] [rbp-F8h] BYREF
  char v29; // [rsp-D8h] [rbp-D8h]
  char v30; // [rsp-D7h] [rbp-D7h]
  __m128i v31[3]; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v32; // [rsp-98h] [rbp-98h] BYREF
  __int16 v33; // [rsp-78h] [rbp-78h]
  __m128i v34[6]; // [rsp-68h] [rbp-68h] BYREF

  if ( a2[1] )
  {
    v3 = (_QWORD *)sub_22077B0(32);
    v4 = v3;
    if ( v3 )
    {
      v5 = (__int64)(v3 + 2);
      v6 = *a2;
      v7 = a2[1];
      v3[1] = 0x100000001LL;
      *v3 = &unk_49E13F0;
      sub_C88F40((__int64)(v3 + 2), v6, v7, 0);
    }
    else
    {
      v5 = 16;
    }
    v8 = (volatile signed __int32 *)a1[1];
    *a1 = v5;
    a1[1] = (__int64)v4;
    if ( v8 )
    {
      if ( &_pthread_key_create )
      {
        v9 = _InterlockedExchangeAdd(v8 + 2, 0xFFFFFFFF);
      }
      else
      {
        v9 = *((_DWORD *)v8 + 2);
        *((_DWORD *)v8 + 2) = v9 - 1;
      }
      if ( v9 == 1 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 16LL))(v8);
        if ( &_pthread_key_create )
        {
          v13 = _InterlockedExchangeAdd(v8 + 3, 0xFFFFFFFF);
        }
        else
        {
          v13 = *((_DWORD *)v8 + 3);
          *((_DWORD *)v8 + 3) = v13 - 1;
        }
        if ( v13 == 1 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 24LL))(v8);
      }
      v5 = *a1;
    }
    v20[1] = 0;
    v20[0] = v21;
    LOBYTE(v21[0]) = 0;
    if ( !(unsigned __int8)sub_C89030((__int64 *)v5, v20) )
    {
      v32.m128i_i64[0] = (__int64)v20;
      v28.m128i_i64[0] = (__int64)"' in -pass-remarks: ";
      v22.m128i_i64[0] = (__int64)"Invalid regular expression '";
      v33 = 260;
      v30 = 1;
      v29 = 3;
      v26 = 260;
      v25.m128i_i64[0] = (__int64)a2;
      v24 = 1;
      v23 = 3;
      sub_9C6370(v27, &v22, &v25, v10, v11, v12);
      sub_9C6370(v31, v27, &v28, v14, v15, v16);
      sub_9C6370(v34, v31, &v32, v17, v18, v19);
      sub_C64D30((__int64)v34, 0);
    }
    if ( (_QWORD *)v20[0] != v21 )
      j_j___libc_free_0(v20[0], v21[0] + 1LL);
  }
}
