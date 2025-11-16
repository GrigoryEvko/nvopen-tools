// Function: sub_30A8030
// Address: 0x30a8030
//
__int64 __fastcall sub_30A8030(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  void *v12; // rdx
  __int64 v13; // rcx
  __int64 j; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 *v17; // rcx
  unsigned __int64 *v18; // r12
  unsigned __int64 *k; // r14
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  _QWORD *v24; // rdi
  __int64 v26; // rdi
  void *v27; // rdx
  __int64 v28; // rdi
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  __int64 v31; // r15
  __int64 i; // r13
  __int64 v33; // rax
  __m128i *v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rax
  __m128i *v37; // rdx
  __int64 v38; // rdi
  __m128i si128; // xmm0
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rdi
  __m128i *v46; // rdx
  __m128i v47; // xmm0
  __int64 v48; // [rsp+8h] [rbp-78h]
  char v50[8]; // [rsp+20h] [rbp-60h] BYREF
  char v51; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v52; // [rsp+30h] [rbp-50h]
  __int64 v53; // [rsp+38h] [rbp-48h]

  v4 = sub_BC0510(a4, &unk_502E1A8, a3);
  if ( *(_QWORD *)(v4 + 48) )
  {
    v5 = v4;
    if ( !*((_DWORD *)a2 + 2) )
    {
      v26 = *a2;
      v27 = *(void **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v27 <= 0xEu )
      {
        sub_CB6200(v26, "Function Info:\n", 0xFu);
      }
      else
      {
        qmemcpy(v27, "Function Info:\n", 15);
        *(_QWORD *)(v26 + 32) += 15LL;
      }
      v31 = *(_QWORD *)(v5 + 128);
      for ( i = v5 + 112; v31 != i; v31 = sub_220EEE0(v31) )
      {
        while ( 1 )
        {
          v42 = sub_CB59D0(*a2, *(_QWORD *)(v31 + 32));
          v43 = *(_QWORD *)(v42 + 32);
          v44 = v42;
          if ( (unsigned __int64)(*(_QWORD *)(v42 + 24) - v43) > 2 )
          {
            *(_BYTE *)(v43 + 2) = 32;
            *(_WORD *)v43 = 14880;
            *(_QWORD *)(v42 + 32) += 3LL;
          }
          else
          {
            v44 = sub_CB6200(v42, (unsigned __int8 *)" : ", 3u);
          }
          v33 = sub_CB6200(v44, *(unsigned __int8 **)(v31 + 48), *(_QWORD *)(v31 + 56));
          v34 = *(__m128i **)(v33 + 32);
          v35 = v33;
          if ( *(_QWORD *)(v33 + 24) - (_QWORD)v34 <= 0xFu )
          {
            v35 = sub_CB6200(v33, ". MaxCounterID: ", 0x10u);
          }
          else
          {
            *v34 = _mm_load_si128((const __m128i *)&xmmword_44CB730);
            *(_QWORD *)(v33 + 32) += 16LL;
          }
          v36 = sub_CB59D0(v35, *(unsigned int *)(v31 + 40));
          v37 = *(__m128i **)(v36 + 32);
          v38 = v36;
          if ( *(_QWORD *)(v36 + 24) - (_QWORD)v37 <= 0x10u )
          {
            v38 = sub_CB6200(v36, ". MaxCallsiteID: ", 0x11u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_44CB740);
            v37[1].m128i_i8[0] = 32;
            *v37 = si128;
            *(_QWORD *)(v36 + 32) += 17LL;
          }
          v40 = sub_CB59D0(v38, *(unsigned int *)(v31 + 44));
          v41 = *(_BYTE **)(v40 + 32);
          if ( *(_BYTE **)(v40 + 24) == v41 )
            break;
          *v41 = 10;
          ++*(_QWORD *)(v40 + 32);
          v31 = sub_220EEE0(v31);
          if ( v31 == i )
            goto LABEL_43;
        }
        sub_CB6200(v40, (unsigned __int8 *)"\n", 1u);
      }
LABEL_43:
      v45 = *a2;
      if ( !*((_DWORD *)a2 + 2) )
      {
        v46 = *(__m128i **)(v45 + 32);
        if ( *(_QWORD *)(v45 + 24) - (_QWORD)v46 <= 0x11u )
        {
          sub_CB6200(v45, "\nCurrent Profile:\n", 0x12u);
        }
        else
        {
          v47 = _mm_load_si128((const __m128i *)&xmmword_44CB750);
          v46[1].m128i_i16[0] = 2618;
          *v46 = v47;
          *(_QWORD *)(v45 + 32) += 18LL;
        }
      }
    }
    v6 = v5 + 8;
    sub_3156B80(*a2, v6);
    v9 = *a2;
    v10 = *(_BYTE **)(*a2 + 32);
    if ( *(_BYTE **)(*a2 + 24) == v10 )
    {
      sub_CB6200(v9, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v9 + 32);
    }
    v48 = a1 + 80;
    if ( *((_DWORD *)a2 + 2) == 1 )
    {
      *(_QWORD *)(a1 + 8) = a1 + 32;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = v48;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)a1 = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
    }
    else
    {
      v11 = *a2;
      v12 = *(void **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v12 <= 0xEu )
      {
        sub_CB6200(v11, "\nFlat Profile:\n", 0xFu);
      }
      else
      {
        v13 = 14949;
        qmemcpy(v12, "\nFlat Profile:\n", 15);
        *(_QWORD *)(v11 + 32) += 15LL;
      }
      sub_30A7FD0((__int64)v50, v6, (__int64)v12, v13, v7, v8);
      for ( j = v53; (char *)j != &v51; j = sub_220EEE0(j) )
      {
        v15 = sub_CB59D0(*a2, *(_QWORD *)(j + 32));
        v16 = *(_QWORD *)(v15 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 2 )
        {
          sub_CB6200(v15, (unsigned __int8 *)" : ", 3u);
        }
        else
        {
          *(_BYTE *)(v16 + 2) = 32;
          *(_WORD *)v16 = 14880;
          *(_QWORD *)(v15 + 32) += 3LL;
        }
        v17 = *(unsigned __int64 **)(j + 40);
        v18 = &v17[*(unsigned int *)(j + 48)];
        for ( k = v17; v18 != k; ++k )
        {
          while ( 1 )
          {
            v20 = sub_CB59D0(*a2, *k);
            v21 = *(_BYTE **)(v20 + 32);
            if ( *(_BYTE **)(v20 + 24) == v21 )
              break;
            ++k;
            *v21 = 32;
            ++*(_QWORD *)(v20 + 32);
            if ( v18 == k )
              goto LABEL_17;
          }
          sub_CB6200(v20, (unsigned __int8 *)" ", 1u);
        }
LABEL_17:
        v22 = *a2;
        v23 = *(_BYTE **)(*a2 + 32);
        if ( *(_BYTE **)(*a2 + 24) == v23 )
        {
          sub_CB6200(v22, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v23 = 10;
          ++*(_QWORD *)(v22 + 32);
        }
      }
      v24 = v52;
      *(_QWORD *)(a1 + 8) = a1 + 32;
      *(_QWORD *)(a1 + 56) = v48;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)a1 = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      sub_30A7180(v24);
    }
  }
  else
  {
    v28 = *a2;
    v29 = *(__m128i **)(*a2 + 32);
    if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v29 <= 0x23u )
    {
      sub_CB6200(v28, "No contextual profile was provided.\n", 0x24u);
    }
    else
    {
      v30 = _mm_load_si128((const __m128i *)&xmmword_44CB710);
      v29[2].m128i_i32[0] = 170812517;
      *v29 = v30;
      v29[1] = _mm_load_si128((const __m128i *)&xmmword_44CB720);
      *(_QWORD *)(v28 + 32) += 36LL;
    }
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
