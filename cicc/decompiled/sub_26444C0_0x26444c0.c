// Function: sub_26444C0
// Address: 0x26444c0
//
void __fastcall sub_26444C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  const char *v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  _DWORD *v13; // rax
  __int64 v14; // rcx
  _DWORD *v15; // rdx
  unsigned int *v16; // r13
  unsigned int *v17; // r12
  unsigned __int64 v18; // rax
  char *v19; // r13
  char *v20; // rbx
  __int64 v21; // rdi
  unsigned int v22; // r12d
  _BYTE *v23; // rax
  char *v24; // [rsp+0h] [rbp-80h] BYREF
  char *v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  _DWORD *v29; // [rsp+30h] [rbp-50h]
  _DWORD *v30; // [rsp+38h] [rbp-48h]
  __m128i v31; // [rsp+40h] [rbp-40h] BYREF
  _DWORD *v32; // [rsp+50h] [rbp-30h]
  _DWORD *v33; // [rsp+58h] [rbp-28h]

  v4 = sub_904010(a2, "Edge from Callee ");
  v5 = sub_CB5A80(v4, *(_QWORD *)a1);
  v6 = sub_904010(v5, " to Caller: ");
  v7 = " (BE)";
  v8 = sub_CB5A80(v6, *(_QWORD *)(a1 + 8));
  if ( !*(_BYTE *)(a1 + 17) )
    v7 = byte_3F871B3;
  v9 = sub_904010(v8, v7);
  v10 = sub_904010(v9, " AllocTypes: ");
  sub_2643CE0(&v31, *(_BYTE *)(a1 + 16));
  sub_CB6200(v10, (unsigned __int8 *)v31.m128i_i64[0], v31.m128i_u64[1]);
  sub_2240A30((unsigned __int64 *)&v31);
  sub_904010(a2, " ContextIds:");
  v13 = *(_DWORD **)(a1 + 32);
  v14 = *(_QWORD *)(a1 + 24);
  v15 = &v13[*(unsigned int *)(a1 + 48)];
  if ( *(_DWORD *)(a1 + 40) )
  {
    for ( ; v15 != v13; ++v13 )
    {
      if ( *v13 <= 0xFFFFFFFD )
        break;
    }
  }
  else
  {
    v13 += *(unsigned int *)(a1 + 48);
  }
  v27 = a1 + 24;
  v28 = v14;
  v29 = v13;
  v30 = v15;
  v31.m128i_i64[0] = a1 + 24;
  v31.m128i_i64[1] = v14;
  v32 = v15;
  v33 = v15;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_2641230((__int64 *)&v24, a1 + 24, (__int64)v15, v14, v11, v12, a1 + 24, v14, v13, v15, a1 + 24, v14, v15);
  v16 = (unsigned int *)v25;
  v17 = (unsigned int *)v24;
  if ( v25 != v24 )
  {
    _BitScanReverse64(&v18, (v25 - v24) >> 2);
    sub_263F8F0(v24, v25, 2LL * (int)(63 - (v18 ^ 0x3F)));
    sub_263F470(v17, v16);
    v17 = (unsigned int *)v24;
    v19 = v25;
    if ( v25 != v24 )
    {
      v20 = v24;
      do
      {
        v22 = *(_DWORD *)v20;
        v23 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v23 )
        {
          v21 = sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
        }
        else
        {
          *v23 = 32;
          v21 = a2;
          ++*(_QWORD *)(a2 + 32);
        }
        v20 += 4;
        sub_CB59D0(v21, v22);
      }
      while ( v19 != v20 );
      v17 = (unsigned int *)v24;
    }
  }
  if ( v17 )
    j_j___libc_free_0((unsigned __int64)v17);
}
