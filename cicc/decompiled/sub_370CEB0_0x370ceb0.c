// Function: sub_370CEB0
// Address: 0x370ceb0
//
__int64 __fastcall sub_370CEB0(__int64 a1, _QWORD *a2, _QWORD *a3, unsigned int a4)
{
  unsigned __int64 v8; // rax
  char *v9; // rsi
  unsigned __int16 v10; // di
  char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19[2]; // [rsp+10h] [rbp-170h] BYREF
  char v20; // [rsp+20h] [rbp-160h] BYREF
  _QWORD v21[4]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v22; // [rsp+50h] [rbp-130h]
  __m128i v23; // [rsp+60h] [rbp-120h] BYREF
  const char *v24; // [rsp+70h] [rbp-110h]
  __int16 v25; // [rsp+80h] [rbp-100h]
  __m128i v26[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+B0h] [rbp-D0h]
  __m128i v28[3]; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v29[2]; // [rsp+F0h] [rbp-90h] BYREF
  char v30; // [rsp+110h] [rbp-70h]
  char v31; // [rsp+111h] [rbp-6Fh]
  __m128i v32; // [rsp+120h] [rbp-60h] BYREF
  char v33; // [rsp+130h] [rbp-50h] BYREF
  _BYTE v34[79]; // [rsp+131h] [rbp-4Fh] BYREF

  if ( a2[9] && !a2[7] && !a2[8] )
  {
    v31 = 1;
    v29[0].m128i_i64[0] = (__int64)")";
    v8 = a4;
    v30 = 3;
    if ( a4 )
    {
      v9 = v34;
      do
      {
        --v9;
        v18 = v8 & 0xF;
        v8 >>= 4;
        *v9 = a0123456789abcd_10[v18];
      }
      while ( v8 );
    }
    else
    {
      v33 = 48;
      v9 = &v33;
    }
    v19[0] = (__int64)&v20;
    sub_370CBD0(v19, v9, (__int64)v34);
    v26[0].m128i_i64[0] = (__int64)v19;
    v27 = 260;
    v10 = 0;
    if ( a3[1] > 3u )
      v10 = *(_WORD *)(*a3 + 2LL);
    v11 = sub_370C640(v10);
    v21[0] = " ";
    v21[2] = v11;
    v21[3] = v12;
    v23.m128i_i64[0] = (__int64)v21;
    v25 = 770;
    v22 = 1283;
    v24 = " (0x";
    sub_9C6370(v28, &v23, v26, 1283, v13, v14);
    sub_9C6370(&v32, v28, v29, v15, (__int64)&v32, v16);
    v17 = a2[9];
    if ( v17 && !a2[7] && !a2[8] && (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL))(v17) )
      (*(void (__fastcall **)(_QWORD, __m128i *))(*(_QWORD *)a2[9] + 32LL))(a2[9], &v32);
    sub_2240A30((unsigned __int64 *)v19);
  }
  (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *))(*a2 + 24LL))(a1, a2, a3);
  return a1;
}
