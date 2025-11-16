// Function: sub_28515F0
// Address: 0x28515f0
//
void __fastcall sub_28515F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rsi
  int v13; // r15d
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // r13
  __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // rdx
  char **v33; // r14
  __int64 v34; // rdi
  char **v35; // rsi
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  char **v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // [rsp-50h] [rbp-50h]
  __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  __int64 v44; // [rsp-40h] [rbp-40h]
  unsigned __int64 v45; // [rsp-40h] [rbp-40h]
  unsigned __int64 v46; // [rsp-40h] [rbp-40h]
  char **v47; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(_QWORD *)a1;
    v43 = a2 + 16;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = v12;
      if ( v12 <= v10 )
      {
        v25 = *(_QWORD *)a1;
        if ( v12 )
        {
          v38 = v9 + 40;
          v39 = (char **)(a2 + 56);
          do
          {
            v40 = v38;
            v47 = v39;
            v38 += 112;
            *(_QWORD *)(v38 - 152) = *(v39 - 5);
            *(_QWORD *)(v38 - 144) = *(v39 - 4);
            *(_BYTE *)(v38 - 136) = *((_BYTE *)v39 - 24);
            *(_BYTE *)(v38 - 128) = *((_BYTE *)v39 - 16);
            *(_QWORD *)(v38 - 120) = *(v39 - 1);
            sub_28502F0(v40, v39, (__int64)v39, v6, v11, a6);
            v39 = v47 + 14;
            *(_QWORD *)(v38 - 64) = v47[6];
            *(_QWORD *)(v38 - 56) = v47[7];
            *(_BYTE *)(v38 - 48) = *((_BYTE *)v47 + 64);
          }
          while ( v9 + 40 + 112 * v12 != v38 );
          v25 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a1 + 8);
          v11 = v9 + 112 * v12;
        }
        v26 = v25 + 112 * v10;
        while ( v11 != v26 )
        {
          v26 -= 112LL;
          v27 = *(_QWORD *)(v26 + 40);
          if ( v27 != v26 + 56 )
          {
            v45 = v11;
            _libc_free(v27);
            v11 = v45;
          }
        }
        *(_DWORD *)(a1 + 8) = v12;
        v28 = *(_QWORD *)a2;
        v29 = *(_QWORD *)a2 + 112LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v29 )
        {
          do
          {
            v29 -= 112;
            v30 = *(_QWORD *)(v29 + 40);
            if ( v30 != v29 + 56 )
              _libc_free(v30);
          }
          while ( v28 != v29 );
        }
      }
      else
      {
        v14 = *(unsigned int *)(a1 + 12);
        if ( v12 > v14 )
        {
          v36 = v9 + 112 * v10;
          while ( v36 != v9 )
          {
            while ( 1 )
            {
              v36 -= 112LL;
              v37 = *(_QWORD *)(v36 + 40);
              if ( v37 == v36 + 56 )
                break;
              _libc_free(v37);
              if ( v36 == v9 )
                goto LABEL_44;
            }
          }
LABEL_44:
          *(_DWORD *)(a1 + 8) = 0;
          sub_2850FC0(a1, v12, v14, v6, v11, a6);
          v12 = *(unsigned int *)(a2 + 8);
          v9 = *(_QWORD *)a1;
          v43 = *(_QWORD *)a2;
          v15 = *(_QWORD *)a2;
          v10 = 0;
        }
        else
        {
          v15 = v43;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v41 = 112 * v10;
            v10 *= 112LL;
            v31 = v9 + 40;
            v32 = v9 + 40 + v10;
            v33 = (char **)(a2 + 56);
            v42 = v32;
            do
            {
              v34 = v31;
              v46 = v10;
              v31 += 112;
              *(_QWORD *)(v31 - 152) = *(v33 - 5);
              *(_QWORD *)(v31 - 144) = *(v33 - 4);
              *(_BYTE *)(v31 - 136) = *((_BYTE *)v33 - 24);
              *(_BYTE *)(v31 - 128) = *((_BYTE *)v33 - 16);
              *(_QWORD *)(v31 - 120) = *(v33 - 1);
              v35 = v33;
              v33 += 14;
              sub_28502F0(v34, v35, v32, v6, v11, a6);
              v10 = v46;
              *(_QWORD *)(v31 - 64) = *(v33 - 8);
              *(_QWORD *)(v31 - 56) = *(v33 - 7);
              *(_BYTE *)(v31 - 48) = *((_BYTE *)v33 - 48);
            }
            while ( v31 != v42 );
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(_QWORD *)a1;
            v43 = *(_QWORD *)a2;
            v15 = *(_QWORD *)a2 + v41;
          }
        }
        v16 = v9 + v10;
        v17 = v15;
        v18 = 112 * v12 + v43;
        if ( v18 != v15 )
        {
          do
          {
            if ( v16 )
            {
              *(_QWORD *)v16 = *(_QWORD *)v17;
              *(__m128i *)(v16 + 8) = _mm_loadu_si128((const __m128i *)(v17 + 8));
              *(_BYTE *)(v16 + 24) = *(_BYTE *)(v17 + 24);
              v19 = *(_QWORD *)(v17 + 32);
              *(_DWORD *)(v16 + 48) = 0;
              *(_QWORD *)(v16 + 32) = v19;
              *(_QWORD *)(v16 + 40) = v16 + 56;
              *(_DWORD *)(v16 + 52) = 4;
              if ( *(_DWORD *)(v17 + 48) )
              {
                v44 = v18;
                sub_28502F0(v16 + 40, (char **)(v17 + 40), v15, v18, v11, a6);
                v18 = v44;
              }
              *(_QWORD *)(v16 + 88) = *(_QWORD *)(v17 + 88);
              *(__m128i *)(v16 + 96) = _mm_loadu_si128((const __m128i *)(v17 + 96));
            }
            v17 += 112;
            v16 += 112LL;
          }
          while ( v18 != v17 );
        }
        *(_DWORD *)(a1 + 8) = v13;
        v20 = *(_QWORD *)a2;
        v21 = *(_QWORD *)a2 + 112LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v21 -= 112;
            v22 = *(_QWORD *)(v21 + 40);
            if ( v22 != v21 + 56 )
              _libc_free(v22);
          }
          while ( v20 != v21 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v23 = v9 + 112 * v10;
      if ( v23 != v9 )
      {
        do
        {
          v23 -= 112LL;
          v24 = *(_QWORD *)(v23 + 40);
          if ( v24 != v23 + 56 )
            _libc_free(v24);
        }
        while ( v23 != v9 );
        v11 = *(_QWORD *)a1;
      }
      if ( v11 != a1 + 16 )
        _libc_free(v11);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v43;
    }
  }
}
