// Function: sub_B827A0
// Address: 0xb827a0
//
__int64 __fastcall sub_B827A0(__int64 *a1, _BYTE *a2, size_t a3)
{
  __int64 v4; // r14
  unsigned int v5; // eax
  unsigned int v6; // r15d
  _QWORD *v7; // rcx
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  _QWORD *v12; // rcx
  _QWORD *v13; // rbx
  __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r12
  __int64 v17; // r12
  __int64 v18; // r12
  __int64 v19; // rax
  _QWORD *v20; // rsi
  _QWORD *v21; // rbx
  _QWORD *v22; // r12
  _QWORD *v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-398h]
  unsigned int v25; // [rsp+1Ch] [rbp-384h]
  __int64 v26; // [rsp+20h] [rbp-380h]
  _QWORD *v27; // [rsp+20h] [rbp-380h]
  _QWORD v28[2]; // [rsp+30h] [rbp-370h] BYREF
  __int64 v29; // [rsp+40h] [rbp-360h] BYREF
  __int64 *v30; // [rsp+50h] [rbp-350h]
  __int64 v31; // [rsp+60h] [rbp-340h] BYREF
  _QWORD v32[2]; // [rsp+80h] [rbp-320h] BYREF
  __int64 v33; // [rsp+90h] [rbp-310h] BYREF
  __int64 *v34; // [rsp+A0h] [rbp-300h]
  __int64 v35; // [rsp+B0h] [rbp-2F0h] BYREF
  __int64 v36[2]; // [rsp+D0h] [rbp-2D0h] BYREF
  __int64 v37; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 *v38; // [rsp+F0h] [rbp-2B0h]
  __int64 v39; // [rsp+100h] [rbp-2A0h] BYREF
  __int64 v40[2]; // [rsp+120h] [rbp-280h] BYREF
  __int64 v41; // [rsp+130h] [rbp-270h] BYREF
  __int64 *v42; // [rsp+140h] [rbp-260h]
  __int64 v43; // [rsp+150h] [rbp-250h] BYREF
  __m128i v44; // [rsp+170h] [rbp-230h] BYREF
  __int64 v45; // [rsp+180h] [rbp-220h] BYREF
  __int64 *v46; // [rsp+190h] [rbp-210h]
  __int64 v47; // [rsp+1A0h] [rbp-200h] BYREF
  _QWORD v48[10]; // [rsp+1C0h] [rbp-1E0h] BYREF
  _QWORD *v49; // [rsp+210h] [rbp-190h]
  unsigned int v50; // [rsp+218h] [rbp-188h]
  _BYTE v51[384]; // [rsp+220h] [rbp-180h] BYREF

  v4 = *a1;
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92740(v4, a2, a3, v5);
  v7 = (_QWORD *)(*(_QWORD *)v4 + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(v4 + 16);
  }
  v27 = v7;
  v11 = sub_C7D670(a3 + 17, 8);
  v12 = v27;
  v13 = (_QWORD *)v11;
  if ( a3 )
  {
    memcpy((void *)(v11 + 16), a2, a3);
    v12 = v27;
  }
  *((_BYTE *)v13 + a3 + 16) = 0;
  *v13 = a3;
  v13[1] = 0;
  *v12 = v13;
  ++*(_DWORD *)(v4 + 12);
  v14 = (__int64 *)(*(_QWORD *)v4 + 8LL * (unsigned int)sub_C929D0(v4, v6));
  v8 = *v14;
  if ( *v14 )
    goto LABEL_10;
  do
  {
    do
    {
      v8 = v14[1];
      ++v14;
    }
    while ( !v8 );
LABEL_10:
    ;
  }
  while ( v8 == -8 );
LABEL_3:
  result = *(unsigned int *)(v8 + 8);
  v10 = *(unsigned int *)(v8 + 12);
  v25 = *(_DWORD *)(v8 + 8);
  v26 = v10 - result;
  if ( v10 != result )
  {
    v15 = a1[2];
    v44 = 0u;
    sub_B17850((__int64)v48, (__int64)"size-info", (__int64)"FunctionIRSizeChange", 20, &v44, v15);
    sub_B16430((__int64)v28, "Pass", 4u, *(_BYTE **)a1[3], *(_QWORD *)(a1[3] + 8));
    v24 = sub_B826F0((__int64)v48, (__int64)v28);
    sub_B18290(v24, ": Function: ", 0xCu);
    sub_B16430((__int64)v32, "Function", 8u, a2, a3);
    v16 = sub_B826F0(v24, (__int64)v32);
    sub_B18290(v16, ": IR instruction count changed from ", 0x24u);
    sub_B169E0(v36, "IRInstrsBefore", 14, v25);
    v17 = sub_B826F0(v16, (__int64)v36);
    sub_B18290(v17, " to ", 4u);
    sub_B169E0(v40, "IRInstrsAfter", 13, v10);
    v18 = sub_B826F0(v17, (__int64)v40);
    sub_B18290(v18, "; Delta: ", 9u);
    sub_B167F0(v44.m128i_i64, "DeltaInstrCount", 15, v26);
    sub_B826F0(v18, (__int64)&v44);
    if ( v46 != &v47 )
      j_j___libc_free_0(v46, v47 + 1);
    if ( (__int64 *)v44.m128i_i64[0] != &v45 )
      j_j___libc_free_0(v44.m128i_i64[0], v45 + 1);
    if ( v42 != &v43 )
      j_j___libc_free_0(v42, v43 + 1);
    if ( (__int64 *)v40[0] != &v41 )
      j_j___libc_free_0(v40[0], v41 + 1);
    if ( v38 != &v39 )
      j_j___libc_free_0(v38, v39 + 1);
    if ( (__int64 *)v36[0] != &v37 )
      j_j___libc_free_0(v36[0], v37 + 1);
    if ( v34 != &v35 )
      j_j___libc_free_0(v34, v35 + 1);
    if ( (__int64 *)v32[0] != &v33 )
      j_j___libc_free_0(v32[0], v33 + 1);
    if ( v30 != &v31 )
      j_j___libc_free_0(v30, v31 + 1);
    if ( (__int64 *)v28[0] != &v29 )
      j_j___libc_free_0(v28[0], v29 + 1);
    v19 = sub_B2BE50(*(_QWORD *)a1[1]);
    v20 = v48;
    sub_B6EB20(v19, (__int64)v48);
    *(_DWORD *)(v8 + 8) = v10;
    v21 = v49;
    v48[0] = &unk_49D9D40;
    v22 = &v49[10 * v50];
    if ( v49 != v22 )
    {
      do
      {
        v22 -= 10;
        v23 = (_QWORD *)v22[4];
        if ( v23 != v22 + 6 )
        {
          v20 = (_QWORD *)(v22[6] + 1LL);
          j_j___libc_free_0(v23, v20);
        }
        if ( (_QWORD *)*v22 != v22 + 2 )
        {
          v20 = (_QWORD *)(v22[2] + 1LL);
          j_j___libc_free_0(*v22, v20);
        }
      }
      while ( v21 != v22 );
      v22 = v49;
    }
    result = (__int64)v51;
    if ( v22 != (_QWORD *)v51 )
      return _libc_free(v22, v20);
  }
  return result;
}
