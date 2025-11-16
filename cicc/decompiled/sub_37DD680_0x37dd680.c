// Function: sub_37DD680
// Address: 0x37dd680
//
__int64 __fastcall sub_37DD680(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // r10
  __int64 v9; // r13
  __int64 v10; // r15
  unsigned int v11; // ecx
  unsigned int *v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // edi
  unsigned int v15; // esi
  __int64 v16; // r9
  int v17; // eax
  __int64 v18; // rdx
  int v19; // esi
  __int64 v20; // r11
  unsigned int v21; // edi
  unsigned int v22; // esi
  unsigned int *v23; // rdx
  __m128i *v24; // rbx
  __m128i *i; // rax
  __int32 v26; // r11d
  __int32 v27; // r9d
  __int32 v28; // ecx
  __int32 v29; // r8d
  __int64 v30; // rdi
  __int64 v31; // rsi
  unsigned int v32; // ecx
  unsigned int v33; // eax
  __int32 v34; // r9d
  __int32 v35; // ecx
  __int32 v36; // edx
  __int32 v37; // eax
  __m128i *v38; // r15
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __m128i v42; // xmm0
  int v43; // eax
  __int64 v44; // r9
  __int64 v45; // rsi
  int v46; // edx
  int v47; // eax
  __int64 v48; // r11
  unsigned int *v49; // [rsp+28h] [rbp-60h]

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 320 )
    return result;
  v7 = a2;
  if ( !a3 )
  {
    v38 = a2;
    goto LABEL_34;
  }
  v9 = a1 + 20;
  v10 = a3;
  v49 = (unsigned int *)(a1 + 40);
  while ( 2 )
  {
    --v10;
    v11 = *(_DWORD *)(a1 + 20);
    v12 = (unsigned int *)(a1 + 20 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (result >> 2)) >> 1));
    v13 = *v12;
    if ( v11 < *v12 || v11 == v13 && *(_DWORD *)(a1 + 24) < v12[1] )
    {
      v14 = v7[-2].m128i_u32[3];
      if ( v13 < v14 || v13 == v14 && v12[1] < v7[-1].m128i_i32[0] )
        goto LABEL_25;
      if ( v11 >= v14 )
      {
        v33 = *(_DWORD *)(a1 + 24);
        if ( v11 != v14 || v33 >= v7[-1].m128i_i32[0] )
        {
          v44 = *(_QWORD *)a1;
          v45 = *(_QWORD *)(a1 + 8);
          *(_DWORD *)(a1 + 4) = v33;
          v46 = *(_DWORD *)(a1 + 16);
          v47 = *(_DWORD *)(a1 + 36);
          v48 = *(_QWORD *)(a1 + 28);
          v21 = *(_DWORD *)a1;
          *(_QWORD *)(a1 + 20) = v44;
          *(_DWORD *)a1 = v11;
          *(_QWORD *)(a1 + 8) = v48;
          *(_DWORD *)(a1 + 16) = v47;
          *(_QWORD *)(a1 + 28) = v45;
          *(_DWORD *)(a1 + 36) = v46;
          v22 = v7[-2].m128i_u32[3];
          goto LABEL_10;
        }
      }
      goto LABEL_29;
    }
    v14 = v7[-2].m128i_u32[3];
    if ( v11 >= v14 )
    {
      if ( v11 == v14 )
      {
        v15 = *(_DWORD *)(a1 + 24);
        if ( v15 < v7[-1].m128i_i32[0] )
          goto LABEL_9;
      }
      if ( v13 >= v14 && (v13 != v14 || v12[1] >= v7[-1].m128i_i32[0]) )
      {
LABEL_25:
        v30 = *(_QWORD *)a1;
        v31 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)a1 = v13;
        v32 = *(_DWORD *)(a1 + 16);
        *(_DWORD *)(a1 + 4) = v12[1];
        *(_DWORD *)(a1 + 8) = v12[2];
        *(_DWORD *)(a1 + 12) = v12[3];
        *(_DWORD *)(a1 + 16) = v12[4];
        *(_QWORD *)v12 = v30;
        *((_QWORD *)v12 + 1) = v31;
        v12[4] = v32;
        v21 = *(_DWORD *)(a1 + 20);
        v11 = *(_DWORD *)a1;
        v22 = v7[-2].m128i_u32[3];
        goto LABEL_10;
      }
LABEL_29:
      v22 = *(_DWORD *)a1;
      v34 = *(_DWORD *)(a1 + 4);
      *(_DWORD *)a1 = v14;
      v35 = *(_DWORD *)(a1 + 8);
      v36 = *(_DWORD *)(a1 + 12);
      v37 = *(_DWORD *)(a1 + 16);
      *(_DWORD *)(a1 + 4) = v7[-1].m128i_i32[0];
      *(_DWORD *)(a1 + 8) = v7[-1].m128i_i32[1];
      *(_DWORD *)(a1 + 12) = v7[-1].m128i_i32[2];
      *(_DWORD *)(a1 + 16) = v7[-1].m128i_i32[3];
      v7[-2].m128i_i32[3] = v22;
      v7[-1].m128i_i32[0] = v34;
      v7[-1].m128i_i32[1] = v35;
      v7[-1].m128i_i32[2] = v36;
      v7[-1].m128i_i32[3] = v37;
      v21 = *(_DWORD *)(a1 + 20);
      v11 = *(_DWORD *)a1;
      goto LABEL_10;
    }
    v15 = *(_DWORD *)(a1 + 24);
LABEL_9:
    v16 = *(_QWORD *)a1;
    v17 = *(_DWORD *)(a1 + 16);
    *(_DWORD *)(a1 + 4) = v15;
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(_DWORD *)(a1 + 36);
    v20 = *(_QWORD *)(a1 + 28);
    v21 = *(_DWORD *)a1;
    *(_QWORD *)(a1 + 20) = v16;
    *(_DWORD *)a1 = v11;
    *(_QWORD *)(a1 + 8) = v20;
    *(_DWORD *)(a1 + 16) = v19;
    *(_QWORD *)(a1 + 28) = v18;
    *(_DWORD *)(a1 + 36) = v17;
    v22 = v7[-2].m128i_u32[3];
LABEL_10:
    v23 = v49;
    v24 = (__m128i *)v9;
    i = v7;
    while ( v21 < v11 || v21 == v11 && *(v23 - 4) < *(_DWORD *)(a1 + 4) )
    {
LABEL_15:
      v21 = *v23;
      v24 = (__m128i *)((char *)v24 + 20);
      v23 += 5;
    }
    for ( i = (__m128i *)((char *)i - 20);
          v22 > v11 || v22 == v11 && *(_DWORD *)(a1 + 4) < i->m128i_i32[1];
          v22 = i->m128i_i32[0] )
    {
      i = (__m128i *)((char *)i - 20);
    }
    if ( v24 < i )
    {
      v26 = *(v23 - 4);
      v27 = *(v23 - 3);
      *(v23 - 5) = v22;
      v28 = *(v23 - 1);
      v29 = *(v23 - 2);
      *(v23 - 4) = i->m128i_u32[1];
      *(v23 - 3) = i->m128i_u32[2];
      *(v23 - 2) = i->m128i_u32[3];
      *(v23 - 1) = i[1].m128i_u32[0];
      v22 = i[-2].m128i_u32[3];
      i->m128i_i32[0] = v21;
      i->m128i_i32[1] = v26;
      i->m128i_i32[2] = v27;
      i->m128i_i32[3] = v29;
      i[1].m128i_i32[0] = v28;
      v11 = *(_DWORD *)a1;
      goto LABEL_15;
    }
    sub_37DD680(v24, v7, v10);
    result = (__int64)v24->m128i_i64 - a1;
    if ( (__int64)v24->m128i_i64 - a1 > 320 )
    {
      if ( v10 )
      {
        v7 = v24;
        continue;
      }
      v38 = v24;
LABEL_34:
      sub_37DD570((char *)a1, v38, (unsigned __int64)v38, a4, a5, a6);
      do
      {
        v42 = _mm_loadu_si128((__m128i *)((char *)v38 - 20));
        v38 = (__m128i *)((char *)v38 - 20);
        v43 = v38[1].m128i_i32[0];
        v38->m128i_i32[0] = *(_DWORD *)a1;
        v38->m128i_i32[1] = *(_DWORD *)(a1 + 4);
        v38->m128i_i32[2] = *(_DWORD *)(a1 + 8);
        v38->m128i_i32[3] = *(_DWORD *)(a1 + 12);
        v38[1].m128i_i32[0] = *(_DWORD *)(a1 + 16);
        result = (__int64)sub_37B6400(
                            a1,
                            0,
                            0xCCCCCCCCCCCCCCCDLL * (((__int64)v38->m128i_i64 - a1) >> 2),
                            v39,
                            v40,
                            v41,
                            v42.m128i_i64[0],
                            v42.m128i_i64[1],
                            v43);
      }
      while ( (__int64)v38->m128i_i64 - a1 > 20 );
    }
    return result;
  }
}
