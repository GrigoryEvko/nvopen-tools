// Function: sub_2E0C490
// Address: 0x2e0c490
//
__int64 __fastcall sub_2E0C490(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        _QWORD *a7,
        unsigned int a8)
{
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 result; // rax
  __int64 v15; // r10
  _QWORD *v16; // rax
  __int64 v17; // r9
  __int64 v18; // r10
  __int64 *v19; // r8
  __int64 *v20; // r15
  _QWORD *v21; // rbx
  int v22; // r12d
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // r12
  unsigned __int64 v32; // r8
  const __m128i *v33; // r14
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rdx
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __m128i *v39; // rax
  int v40; // edi
  char *v41; // r14
  __int64 *v42; // rax
  __int64 v43; // [rsp+0h] [rbp-E0h]
  __int64 v44; // [rsp+8h] [rbp-D8h]
  __int64 v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+20h] [rbp-C0h]
  __int64 v48; // [rsp+20h] [rbp-C0h]
  const void *v49; // [rsp+30h] [rbp-B0h]
  __int64 v50; // [rsp+30h] [rbp-B0h]
  const void *v51; // [rsp+40h] [rbp-A0h]
  __int64 *v53; // [rsp+58h] [rbp-88h]
  __int128 v54; // [rsp+60h] [rbp-80h]
  __int64 *v56; // [rsp+78h] [rbp-68h]
  __int64 v57; // [rsp+78h] [rbp-68h]
  _QWORD v60[10]; // [rsp+90h] [rbp-50h] BYREF

  v53 = a2;
  v8 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)&v54 = a4;
  for ( *((_QWORD *)&v54 + 1) = a3; v8; v8 = *(_QWORD *)(v8 + 104) )
  {
    v9 = *(_QWORD *)(v8 + 112);
    v10 = *(_QWORD *)(v8 + 120);
    v11 = v9 & a3;
    v12 = v10 & a4;
    v13 = v10 & a4 | v9 & a3;
    if ( v13 )
    {
      if ( v9 == v11 && v10 == v12 )
      {
        v15 = v8;
      }
      else
      {
        *(_QWORD *)(v8 + 112) = ~v11 & v9;
        *(_QWORD *)(v8 + 120) = ~v12 & v10;
        v16 = (_QWORD *)sub_A777F0(0x80u, v53);
        v18 = (__int64)v16;
        if ( v16 )
        {
          v16[12] = 0;
          v51 = v16 + 2;
          *v16 = v16 + 2;
          v16[1] = 0x200000000LL;
          v49 = v16 + 10;
          v16[8] = v16 + 10;
          v16[9] = 0x200000000LL;
          if ( (_QWORD *)v8 != v16 )
          {
            v19 = *(__int64 **)(v8 + 64);
            v47 = (__int64)(v16 + 8);
            v56 = &v19[*(unsigned int *)(v8 + 72)];
            if ( v19 != v56 )
            {
              v46 = v8;
              v20 = *(__int64 **)(v8 + 64);
              v45 = v11;
              v21 = v16;
              v44 = v12;
              v22 = 0;
              do
              {
                v23 = *v20;
                v24 = sub_A777F0(0x10u, v53);
                if ( v24 )
                {
                  v26 = *(_QWORD *)(v23 + 8);
                  *(_DWORD *)v24 = v22;
                  *(_QWORD *)(v24 + 8) = v26;
                }
                v27 = *((unsigned int *)v21 + 18);
                v17 = v27 + 1;
                if ( v27 + 1 > (unsigned __int64)*((unsigned int *)v21 + 19) )
                {
                  v43 = v24;
                  sub_C8D5F0(v47, v49, v27 + 1, 8u, v25, v17);
                  v27 = *((unsigned int *)v21 + 18);
                  v24 = v43;
                }
                ++v20;
                *(_QWORD *)(v21[8] + 8 * v27) = v24;
                v22 = *((_DWORD *)v21 + 18) + 1;
                *((_DWORD *)v21 + 18) = v22;
              }
              while ( v56 != v20 );
              v18 = (__int64)v21;
              v8 = v46;
              v11 = v45;
              v12 = v44;
            }
            if ( *(_QWORD *)v8 != *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8) )
            {
              v28 = *(unsigned int *)(v18 + 8);
              v29 = v18;
              v50 = v11;
              v30 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
              v48 = v12;
              v31 = *(_QWORD *)v8;
              do
              {
                v32 = v28 + 1;
                v33 = (const __m128i *)v60;
                v34 = **(unsigned int **)(v31 + 16);
                v35 = *(_QWORD *)(v29 + 64);
                v60[0] = *(_QWORD *)v31;
                v36 = *(_QWORD *)(v35 + 8 * v34);
                v60[1] = *(_QWORD *)(v31 + 8);
                v37 = *(unsigned int *)(v29 + 12);
                v60[2] = v36;
                v38 = *(_QWORD *)v29;
                if ( v28 + 1 > v37 )
                {
                  if ( v38 > (unsigned __int64)v60 || (unsigned __int64)v60 >= v38 + 24 * v28 )
                  {
                    v33 = (const __m128i *)v60;
                    sub_C8D5F0(v29, v51, v32, 0x18u, v32, v17);
                    v38 = *(_QWORD *)v29;
                    v28 = *(unsigned int *)(v29 + 8);
                  }
                  else
                  {
                    v41 = (char *)v60 - v38;
                    sub_C8D5F0(v29, v51, v32, 0x18u, v32, v17);
                    v38 = *(_QWORD *)v29;
                    v28 = *(unsigned int *)(v29 + 8);
                    v33 = (const __m128i *)&v41[*(_QWORD *)v29];
                  }
                }
                v31 += 24;
                v39 = (__m128i *)(v38 + 24 * v28);
                *v39 = _mm_loadu_si128(v33);
                v39[1].m128i_i64[0] = v33[1].m128i_i64[0];
                v28 = (unsigned int)(*(_DWORD *)(v29 + 8) + 1);
                *(_DWORD *)(v29 + 8) = v28;
              }
              while ( v30 != v31 );
              v11 = v50;
              v18 = v29;
              v12 = v48;
            }
          }
          *(_QWORD *)(v18 + 104) = 0;
          *(_QWORD *)(v18 + 112) = v11;
          *(_QWORD *)(v18 + 120) = v12;
        }
        v57 = v18;
        *(_QWORD *)(v18 + 104) = *(_QWORD *)(a1 + 104);
        v40 = *(_DWORD *)(a1 + 112);
        *(_QWORD *)(a1 + 104) = v18;
        sub_2E0A7D0(v40, v18, v11, v12, a7, a8);
        v13 = *(unsigned int *)(a1 + 112);
        a2 = (__int64 *)v8;
        sub_2E0A7D0(v13, v8, *(_QWORD *)(v8 + 112), *(_QWORD *)(v8 + 120), a7, a8);
        v15 = v57;
      }
      if ( !*(_QWORD *)(a5 + 16) )
        goto LABEL_36;
      a2 = (__int64 *)v15;
      (*(void (__fastcall **)(unsigned __int64, __int64))(a5 + 24))(a5, v15);
      *((_QWORD *)&v54 + 1) &= ~v11;
      *(_QWORD *)&v54 = ~v12 & v54;
    }
  }
  result = v54 | *((_QWORD *)&v54 + 1);
  if ( v54 != 0 )
  {
    v42 = (__int64 *)sub_A777F0(0x80u, v53);
    a2 = v42;
    if ( v42 )
    {
      v10 = (unsigned __int64)(v42 + 10);
      v42[12] = 0;
      *v42 = (__int64)(v42 + 2);
      v42[1] = 0x200000000LL;
      v42[9] = 0x200000000LL;
      v42[8] = (__int64)(v42 + 10);
      v42[14] = *((_QWORD *)&v54 + 1);
      v42[13] = 0;
      v42[15] = v54;
    }
    v42[13] = *(_QWORD *)(a1 + 104);
    *(_QWORD *)(a1 + 104) = v42;
    v13 = a5;
    if ( !*(_QWORD *)(a5 + 16) )
LABEL_36:
      sub_4263D6(v13, a2, v10);
    return (*(__int64 (**)(void))(a5 + 24))();
  }
  return result;
}
