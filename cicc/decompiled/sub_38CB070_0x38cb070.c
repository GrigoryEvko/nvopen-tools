// Function: sub_38CB070
// Address: 0x38cb070
//
void __fastcall sub_38CB070(_QWORD *a1, unsigned int *a2)
{
  __int64 v3; // rdi
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned int v10; // edi
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 *v15; // rax
  __m128i *v16; // rsi
  unsigned int v17; // [rsp+4h] [rbp-4Ch] BYREF
  unsigned int *v18; // [rsp+8h] [rbp-48h] BYREF
  __m128i v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+20h] [rbp-30h]

  v3 = a1[1];
  if ( *(_BYTE *)(v3 + 1040) )
  {
    v5 = sub_38BFA60(v3, 1);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 176LL))(a1, v5, 0);
    v6 = a1[1];
    v20 = v5;
    *(_BYTE *)(v6 + 1040) = 0;
    v7 = a1[1];
    v8 = *(_QWORD *)(v6 + 1024);
    v9 = *(_QWORD *)(v6 + 1032);
    v10 = *(_DWORD *)(v7 + 1164);
    v11 = *(_QWORD *)(v7 + 992);
    v19.m128i_i64[0] = v8;
    v12 = v7 + 984;
    v19.m128i_i64[1] = v9;
    v17 = v10;
    if ( !v11 )
      goto LABEL_9;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        v14 = *(_QWORD *)(v11 + 24);
        if ( v10 <= *(_DWORD *)(v11 + 32) )
          break;
        v11 = *(_QWORD *)(v11 + 24);
        if ( !v14 )
          goto LABEL_7;
      }
      v12 = v11;
      v11 = *(_QWORD *)(v11 + 16);
    }
    while ( v13 );
LABEL_7:
    if ( v7 + 984 == v12 || v10 < *(_DWORD *)(v12 + 32) )
    {
LABEL_9:
      v18 = &v17;
      v12 = sub_38C9BD0((_QWORD *)(v7 + 976), v12, &v18);
    }
    v18 = a2;
    v15 = (unsigned __int64 *)sub_38CAD90(v12 + 536, (__int64 *)&v18);
    v16 = (__m128i *)v15[1];
    if ( v16 == (__m128i *)v15[2] )
    {
      sub_38C82D0(v15, v16, &v19);
    }
    else
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128(&v19);
        v16[1].m128i_i64[0] = v20;
        v16 = (__m128i *)v15[1];
      }
      v15[1] = (unsigned __int64)&v16[1].m128i_u64[1];
    }
  }
}
