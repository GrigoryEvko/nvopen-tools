// Function: sub_EA31A0
// Address: 0xea31a0
//
void __fastcall sub_EA31A0(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // r8
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rdi
  bool v15; // cc
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // r15
  unsigned int v23; // eax
  const void **v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // r15
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rbx
  unsigned int v35; // eax
  unsigned __int64 v36; // [rsp-48h] [rbp-48h]
  __int64 v37; // [rsp-48h] [rbp-48h]
  __int64 *v38; // [rsp-48h] [rbp-48h]
  __int64 v39; // [rsp-40h] [rbp-40h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  __int64 v41; // [rsp-40h] [rbp-40h]
  unsigned __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a2;
    v4 = a2[1];
    v5 = *a2;
    v6 = *a1;
    v7 = v4 - *a2;
    if ( a1[2] - *a1 >= (unsigned __int64)v7 )
    {
      v8 = a1[1];
      v9 = v8 - v6;
      v10 = v8 - v6;
      if ( v7 > (unsigned __int64)(v8 - v6) )
      {
        v28 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 3);
        if ( v9 > 0 )
        {
          v29 = v6 + 24;
          v30 = v5 + 24;
          do
          {
            while ( 1 )
            {
              v15 = *(_DWORD *)(v29 + 8) <= 0x40u;
              *(_DWORD *)(v29 - 24) = *(_DWORD *)(v30 - 24);
              *(__m128i *)(v29 - 16) = _mm_loadu_si128((const __m128i *)(v30 - 16));
              if ( v15 && *(_DWORD *)(v30 + 8) <= 0x40u )
                break;
              v31 = v30;
              v32 = v29;
              v38 = v2;
              v29 += 40;
              v42 = v28;
              v30 += 40;
              sub_C43990(v32, v31);
              v2 = v38;
              v28 = v42 - 1;
              if ( v42 == 1 )
                goto LABEL_44;
            }
            v33 = *(_QWORD *)v30;
            v29 += 40;
            v30 += 40;
            *(_QWORD *)(v29 - 40) = v33;
            *(_DWORD *)(v29 - 32) = *(_DWORD *)(v30 - 32);
            --v28;
          }
          while ( v28 );
LABEL_44:
          v8 = a1[1];
          v6 = *a1;
          v4 = v2[1];
          v5 = *v2;
          v10 = v8 - *a1;
        }
        v34 = v5 + v10;
        if ( v34 == v4 )
        {
          v18 = v6 + v7;
          goto LABEL_17;
        }
        do
        {
          if ( v8 )
          {
            *(_DWORD *)v8 = *(_DWORD *)v34;
            *(__m128i *)(v8 + 8) = _mm_loadu_si128((const __m128i *)(v34 + 8));
            v35 = *(_DWORD *)(v34 + 32);
            *(_DWORD *)(v8 + 32) = v35;
            if ( v35 <= 0x40 )
            {
              *(_QWORD *)(v8 + 24) = *(_QWORD *)(v34 + 24);
            }
            else
            {
              v43 = v4;
              sub_C43780(v8 + 24, (const void **)(v34 + 24));
              v4 = v43;
            }
          }
          v34 += 40;
          v8 += 40;
        }
        while ( v4 != v34 );
      }
      else
      {
        if ( v7 <= 0 )
          goto LABEL_15;
        v11 = v6 + 24;
        v12 = v5 + 24;
        v13 = 0xCCCCCCCCCCCCCCCDLL * (v7 >> 3);
        do
        {
          while ( 1 )
          {
            v15 = *(_DWORD *)(v11 + 8) <= 0x40u;
            *(_DWORD *)(v11 - 24) = *(_DWORD *)(v12 - 24);
            *(__m128i *)(v11 - 16) = _mm_loadu_si128((const __m128i *)(v12 - 16));
            if ( v15 && *(_DWORD *)(v12 + 8) <= 0x40u )
              break;
            v14 = v11;
            v36 = v13;
            v11 += 40;
            v39 = v12;
            sub_C43990(v14, v12);
            v12 = v39 + 40;
            v13 = v36 - 1;
            if ( v36 == 1 )
              goto LABEL_10;
          }
          v16 = *(_QWORD *)v12;
          v11 += 40;
          v12 += 40;
          *(_QWORD *)(v11 - 40) = v16;
          *(_DWORD *)(v11 - 32) = *(_DWORD *)(v12 - 32);
          --v13;
        }
        while ( v13 );
LABEL_10:
        v6 += v7;
        while ( v8 != v6 )
        {
          if ( *(_DWORD *)(v6 + 32) > 0x40u )
          {
            v17 = *(_QWORD *)(v6 + 24);
            if ( v17 )
              j_j___libc_free_0_0(v17);
          }
          v6 += 40;
LABEL_15:
          ;
        }
      }
      v18 = *a1 + v7;
LABEL_17:
      a1[1] = v18;
      return;
    }
    if ( v7 )
    {
      if ( (unsigned __int64)v7 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(a1, a2, v5);
      v37 = *a2;
      v40 = a2[1];
      v19 = sub_22077B0(v40 - *a2);
      v4 = v40;
      v5 = v37;
      v20 = v19;
    }
    else
    {
      v20 = 0;
    }
    v21 = v5;
    v22 = v20;
    if ( v4 == v5 )
    {
LABEL_28:
      v25 = a1[1];
      v26 = *a1;
      if ( v25 != *a1 )
      {
        do
        {
          if ( *(_DWORD *)(v26 + 32) > 0x40u )
          {
            v27 = *(_QWORD *)(v26 + 24);
            if ( v27 )
              j_j___libc_free_0_0(v27);
          }
          v26 += 40;
        }
        while ( v25 != v26 );
        v26 = *a1;
      }
      if ( v26 )
        j_j___libc_free_0(v26, a1[2] - v26);
      v18 = v20 + v7;
      *a1 = v20;
      a1[2] = v18;
      goto LABEL_17;
    }
    while ( 1 )
    {
      if ( !v22 )
        goto LABEL_24;
      *(_DWORD *)v22 = *(_DWORD *)v21;
      *(__m128i *)(v22 + 8) = _mm_loadu_si128((const __m128i *)(v21 + 8));
      v23 = *(_DWORD *)(v21 + 32);
      *(_DWORD *)(v22 + 32) = v23;
      if ( v23 <= 0x40 )
      {
        *(_QWORD *)(v22 + 24) = *(_QWORD *)(v21 + 24);
LABEL_24:
        v21 += 40;
        v22 += 40;
        if ( v4 == v21 )
          goto LABEL_28;
      }
      else
      {
        v24 = (const void **)(v21 + 24);
        v41 = v4;
        v21 += 40;
        sub_C43780(v22 + 24, v24);
        v4 = v41;
        v22 += 40;
        if ( v41 == v21 )
          goto LABEL_28;
      }
    }
  }
}
