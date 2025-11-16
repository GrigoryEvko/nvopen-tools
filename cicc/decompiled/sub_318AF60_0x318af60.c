// Function: sub_318AF60
// Address: 0x318af60
//
__int64 __fastcall sub_318AF60(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v9; // rax
  char *v10; // r12
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __m128i v22; // xmm2
  __int64 v23; // rax
  __m128i v24; // xmm0
  __int64 v25; // rax
  __int64 v26; // rdi
  char *v27; // r12
  __int64 v28; // [rsp+0h] [rbp-B0h] BYREF
  int v29; // [rsp+8h] [rbp-A8h] BYREF
  __m128i v30; // [rsp+10h] [rbp-A0h] BYREF
  void (__fastcall *v31)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-90h]
  __int64 v32; // [rsp+28h] [rbp-88h]
  char v33[16]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v34; // [rsp+40h] [rbp-70h]
  char v35; // [rsp+50h] [rbp-60h]
  __int64 v36; // [rsp+60h] [rbp-50h] BYREF
  __m128i v37; // [rsp+68h] [rbp-48h] BYREF
  void (__fastcall *v38)(__m128i *, __m128i *, __int64); // [rsp+78h] [rbp-38h]
  __int64 v39; // [rsp+80h] [rbp-30h]

  v3 = *a2;
  v29 = 0;
  v28 = v3;
  sub_318A950((__int64)v33, a1, &v28, &v29);
  v6 = v34;
  if ( v35 )
  {
    v9 = *a2;
    v38 = 0;
    v10 = (char *)&v36;
    v11 = _mm_loadu_si128(&v30);
    v12 = _mm_loadu_si128(&v37);
    v31 = 0;
    v13 = *(unsigned int *)(a1 + 44);
    v14 = *(_QWORD *)(a1 + 32);
    v36 = v9;
    v15 = v32;
    v32 = v39;
    v16 = *(unsigned int *)(a1 + 40);
    v30 = v12;
    v39 = v15;
    v17 = v16 + 1;
    v37 = v11;
    v18 = v16;
    if ( v16 + 1 > v13 )
    {
      v26 = a1 + 32;
      if ( v14 > (unsigned __int64)&v36 || (unsigned __int64)&v36 >= v14 + 40 * v16 )
      {
        sub_3187AB0(v26, v17, v16, v14, v4, v5);
        v16 = *(unsigned int *)(a1 + 40);
        v14 = *(_QWORD *)(a1 + 32);
        v18 = *(_DWORD *)(a1 + 40);
      }
      else
      {
        v27 = (char *)&v36 - v14;
        sub_3187AB0(v26, v17, v16, v14, v4, v5);
        v14 = *(_QWORD *)(a1 + 32);
        v16 = *(unsigned int *)(a1 + 40);
        v10 = &v27[v14];
        v18 = *(_DWORD *)(a1 + 40);
      }
    }
    v19 = v14 + 40 * v16;
    if ( v19 )
    {
      v20 = *(_QWORD *)v10;
      v21 = *(_QWORD *)(v19 + 32);
      *(_QWORD *)(v19 + 24) = 0;
      v22 = _mm_loadu_si128((const __m128i *)(v19 + 8));
      *(_QWORD *)v19 = v20;
      v23 = *((_QWORD *)v10 + 3);
      *((_QWORD *)v10 + 3) = 0;
      v24 = _mm_loadu_si128((const __m128i *)(v10 + 8));
      *(_QWORD *)(v19 + 24) = v23;
      v25 = *((_QWORD *)v10 + 4);
      *(__m128i *)(v10 + 8) = v22;
      *((_QWORD *)v10 + 4) = v21;
      *(_QWORD *)(v19 + 32) = v25;
      *(__m128i *)(v19 + 8) = v24;
      v18 = *(_DWORD *)(a1 + 40);
    }
    *(_DWORD *)(a1 + 40) = v18 + 1;
    if ( v38 )
      v38(&v37, &v37, 3);
    if ( v31 )
      v31(&v30, &v30, 3);
    v7 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
    *(_DWORD *)(v6 + 8) = v7;
  }
  else
  {
    v7 = *(unsigned int *)(v34 + 8);
  }
  return *(_QWORD *)(a1 + 32) + 40 * v7 + 8;
}
