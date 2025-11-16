// Function: sub_3268760
// Address: 0x3268760
//
__int64 __fastcall sub_3268760(__int64 a1, __int64 a2, __int32 a3, __int64 a4)
{
  _BYTE *v7; // r14
  __int64 v8; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r15
  const __m128i *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // edx
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int128 v26; // [rsp-20h] [rbp-E0h]
  __int128 v27; // [rsp-10h] [rbp-D0h]
  __int64 v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+10h] [rbp-B0h]
  int v31; // [rsp+18h] [rbp-A8h]
  __int64 v32; // [rsp+18h] [rbp-A8h]
  unsigned int v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+20h] [rbp-A0h]
  __int64 v36; // [rsp+28h] [rbp-98h]
  __int64 v37; // [rsp+28h] [rbp-98h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  __int64 v39; // [rsp+38h] [rbp-88h]
  __int64 v40; // [rsp+40h] [rbp-80h] BYREF
  int v41; // [rsp+48h] [rbp-78h]
  __m128i v42; // [rsp+50h] [rbp-70h]
  __m128i v43; // [rsp+60h] [rbp-60h]
  __m128i v44; // [rsp+70h] [rbp-50h]
  __m128i v45; // [rsp+80h] [rbp-40h]

  v7 = *(_BYTE **)a1;
  v36 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(a4 + 24)) )
  {
    v40 = sub_33CB280(*(unsigned int *)(a4 + 24), ((unsigned __int8)(*(_DWORD *)(a4 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v40) || (_DWORD)v40 != 98 )
      return 0;
    v33 = *(_DWORD *)(a4 + 24);
    v10 = sub_33CB160(v33);
    v11 = v33;
    if ( BYTE4(v10) )
    {
      v12 = *(_QWORD *)(a4 + 40) + 40LL * (unsigned int)v10;
      if ( *(_QWORD *)v12 != *(_QWORD *)(v36 + 16) || *(_DWORD *)(v12 + 8) != *(_DWORD *)(v36 + 24) )
      {
        v13 = sub_33D1720(*(_QWORD *)v12, 0);
        v11 = v33;
        if ( !v13 )
          return 0;
      }
    }
    v39 = sub_33CB1F0(v11);
    if ( BYTE4(v39) )
    {
      v14 = *(_QWORD *)(a4 + 40) + 40LL * (unsigned int)v39;
      if ( *(_QWORD *)(v36 + 32) != *(_QWORD *)v14 || *(_DWORD *)(v36 + 40) != *(_DWORD *)(v14 + 8) )
        return 0;
    }
  }
  else if ( *(_DWORD *)(a4 + 24) != 98 )
  {
    return 0;
  }
  if ( *v7 )
  {
    if ( !**(_BYTE **)(a1 + 8) )
    {
LABEL_5:
      v8 = *(_QWORD *)(a4 + 56);
      if ( v8 && !*(_QWORD *)(v8 + 32) )
        goto LABEL_20;
      return 0;
    }
  }
  else
  {
    if ( (*(_BYTE *)(a4 + 29) & 2) == 0 )
      return 0;
    if ( !**(_BYTE **)(a1 + 8) )
      goto LABEL_5;
  }
LABEL_20:
  v15 = *(__int64 **)(a1 + 40);
  v16 = *(_QWORD *)(a4 + 40);
  v17 = *(const __m128i **)(a1 + 16);
  v28 = *(_QWORD *)(a1 + 32);
  v37 = *v15;
  v31 = *(_DWORD *)(v16 + 8);
  v30 = *(_QWORD *)v16;
  v34 = v15[1];
  v18 = sub_33CB7C0(244);
  v40 = v18;
  v19 = v17->m128i_i64[0];
  v40 = v30;
  v41 = v31;
  v42 = _mm_loadu_si128(v17 + 1);
  *((_QWORD *)&v27 + 1) = 3;
  *(_QWORD *)&v27 = &v40;
  v43 = _mm_loadu_si128(v17 + 2);
  v29 = sub_33FC220(v19, v18, v28, v37, v34, (unsigned int)&v40, v27);
  LODWORD(v30) = v20;
  v21 = **(unsigned int **)(a1 + 24);
  v22 = *(__int64 **)(a1 + 40);
  v32 = *(_QWORD *)(a1 + 32);
  v38 = *v22;
  v23 = *(_QWORD *)(v16 + 40);
  v35 = v22[1];
  LODWORD(v16) = *(_DWORD *)(v16 + 48);
  v24 = sub_33CB7C0(v21);
  v40 = v24;
  v25 = v17->m128i_i64[0];
  v40 = v29;
  v41 = v30;
  v42.m128i_i64[0] = v23;
  v42.m128i_i32[2] = v16;
  v43.m128i_i64[0] = a2;
  v43.m128i_i32[2] = a3;
  v44 = _mm_loadu_si128(v17 + 1);
  *((_QWORD *)&v26 + 1) = 5;
  *(_QWORD *)&v26 = &v40;
  v45 = _mm_loadu_si128(v17 + 2);
  return sub_33FC220(v25, v24, v32, v38, v35, (unsigned int)&v40, v26);
}
