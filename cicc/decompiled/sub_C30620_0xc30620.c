// Function: sub_C30620
// Address: 0xc30620
//
__int64 __fastcall sub_C30620(__int64 *a1, __m128i *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v7; // rdi
  _BYTE *v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  __m128i v15; // rax
  void (__fastcall *v16)(__int64 *, __m128i *); // rcx
  __int64 v17; // rax
  char v18; // [rsp+7h] [rbp-B9h] BYREF
  __int64 v19; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD v20[2]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v21[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v22; // [rsp+30h] [rbp-90h] BYREF
  __int64 v23; // [rsp+38h] [rbp-88h]
  _QWORD v24[2]; // [rsp+40h] [rbp-80h] BYREF
  __m128i v25; // [rsp+50h] [rbp-70h] BYREF
  __int64 v26; // [rsp+60h] [rbp-60h]
  __int64 v27; // [rsp+68h] [rbp-58h]
  __int64 v28; // [rsp+70h] [rbp-50h]
  __int64 v29; // [rsp+78h] [rbp-48h]
  _QWORD *v30; // [rsp+80h] [rbp-40h]

  v4 = sub_CB0A70(a1);
  if ( *(_DWORD *)(v4 + 8) == 2 )
  {
    sub_F02AE0(&v25, v4 + 32, a2[1].m128i_i64[0], a2[1].m128i_i64[1]);
    v5 = a2->m128i_i64[0];
    LODWORD(v22) = v25.m128i_i32[0];
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _QWORD *, __m128i *))(*a1 + 120))(
           a1,
           v5,
           1,
           0,
           v21,
           &v25) )
    {
      sub_C2F5F0(a1, (__int64)&v22);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25.m128i_i64[0]);
    }
  }
  else
  {
    v7 = a2[1].m128i_i64[1];
    if ( !v7 )
      goto LABEL_20;
    v8 = (_BYTE *)a2[1].m128i_i64[0];
    v9 = 0;
    do
    {
      v10 = *v8++ == 10;
      v9 += v10;
    }
    while ( (_BYTE *)(v7 + a2[1].m128i_i64[0]) != v8 );
    if ( v9 > 1 )
    {
      v11 = *a1;
      v12 = a2->m128i_i64[0];
      v20[0] = a2[1].m128i_i64[0];
      v20[1] = v7;
      if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, char *, __int64 *))(v11 + 120))(
             a1,
             v12,
             1,
             0,
             &v18,
             &v19) )
      {
        if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
        {
          LOBYTE(v24[0]) = 0;
          v29 = 0x100000000LL;
          v30 = &v22;
          v22 = v24;
          v23 = 0;
          v25.m128i_i64[1] = 0;
          v26 = 0;
          v27 = 0;
          v28 = 0;
          v25.m128i_i64[0] = (__int64)&unk_49DD210;
          sub_CB5980(&v25, 0, 0, 0);
          v17 = sub_CB0A70(a1);
          sub_CB2AD0(v20, v17, &v25);
          v21[0] = v22;
          v21[1] = v23;
          (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 224))(a1, v21);
          v25.m128i_i64[0] = (__int64)&unk_49DD210;
          sub_CB5840(&v25);
          if ( v22 != v24 )
            j_j___libc_free_0(v22, v24[0] + 1LL);
        }
        else
        {
          v13 = *a1;
          v22 = 0;
          v23 = 0;
          (*(void (__fastcall **)(__int64 *, _QWORD **))(v13 + 224))(a1, &v22);
          v14 = sub_CB0A70(a1);
          v15.m128i_i64[0] = sub_CB2B30(v22, v23, v14, v20);
          if ( v15.m128i_i64[1] )
          {
            v16 = *(void (__fastcall **)(__int64 *, __m128i *))(*a1 + 248);
            LOWORD(v28) = 261;
            v25 = v15;
            v16(a1, &v25);
          }
        }
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v19);
      }
    }
    else
    {
LABEL_20:
      if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _QWORD **, __m128i *))(*a1 + 120))(
             a1,
             a2->m128i_i64[0],
             1,
             0,
             &v22,
             &v25) )
      {
        sub_C300B0(a1, (__int64)a2[1].m128i_i64);
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25.m128i_i64[0]);
      }
    }
  }
  v27 = 0;
  return sub_C302C0(a1, (__int64)"DebugLoc", a2 + 2, &v25, 0);
}
