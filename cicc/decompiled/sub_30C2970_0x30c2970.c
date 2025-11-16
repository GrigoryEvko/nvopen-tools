// Function: sub_30C2970
// Address: 0x30c2970
//
__int64 __fastcall sub_30C2970(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 i; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  const __m128i *v19; // r15
  unsigned __int64 v20; // r8
  __m128i *v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdx
  char *v26; // rax
  char **v27; // r9
  char *v28; // rdx
  __int64 v30; // rdi
  char *v31; // r15
  const void *v32; // [rsp+0h] [rbp-190h]
  __int64 v33; // [rsp+28h] [rbp-168h] BYREF
  __int64 v34; // [rsp+30h] [rbp-160h] BYREF
  int v35; // [rsp+38h] [rbp-158h]
  int v36; // [rsp+3Ch] [rbp-154h]
  __int64 v37; // [rsp+40h] [rbp-150h]
  _BYTE v38[32]; // [rsp+50h] [rbp-140h] BYREF
  __int16 v39; // [rsp+70h] [rbp-120h]
  unsigned __int64 v40[4]; // [rsp+80h] [rbp-110h] BYREF
  __int16 v41; // [rsp+A0h] [rbp-F0h]
  char *v42[4]; // [rsp+B0h] [rbp-E0h] BYREF
  __int16 v43; // [rsp+D0h] [rbp-C0h]
  __int64 *v44; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+F0h] [rbp-A0h] BYREF
  int v46; // [rsp+110h] [rbp-80h]
  __int64 *v47; // [rsp+120h] [rbp-70h] BYREF
  __int64 v48; // [rsp+128h] [rbp-68h]
  _BYTE v49[16]; // [rsp+130h] [rbp-60h] BYREF
  __int16 v50; // [rsp+140h] [rbp-50h]

  *(_QWORD *)(a1 + 56) = a1 + 72;
  v32 = (const void *)(a1 + 72);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 0x200000000LL;
  *(_QWORD *)a1 = sub_CC7DD0(a2 + 232);
  *(_QWORD *)(a1 + 8) = v3;
  v4 = sub_CC78E0(a2 + 232);
  *(_QWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 16) = v4;
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a2 + 280);
  v6 = sub_BA8DC0(a2, (__int64)"dx.valver", 9);
  if ( v6 )
  {
    v7 = sub_B91A10(v6, 0);
    v8 = *(_BYTE *)(v7 - 16);
    if ( (v8 & 2) != 0 )
    {
      v11 = *(_QWORD **)(v7 - 32);
      v12 = *(_QWORD *)(*v11 + 136LL);
    }
    else
    {
      v9 = v7 - 8LL * ((v8 >> 2) & 0xF);
      v10 = *(_QWORD *)(v9 - 16);
      v11 = (_QWORD *)(v9 - 16);
      v12 = *(_QWORD *)(v10 + 136);
    }
    v13 = *(_QWORD *)(v11[1] + 136LL);
    v14 = *(_QWORD **)(v13 + 24);
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
      v14 = (_QWORD *)*v14;
    v15 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v15 = (_QWORD *)*v15;
    *(_DWORD *)(a1 + 36) = (_DWORD)v15;
    *(_DWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 40) = (unsigned int)v14 & 0x7FFFFFFF | 0x80000000LL;
  }
  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    v23 = i - 56;
    if ( !i )
      v23 = 0;
    if ( (unsigned __int8)sub_B2D620(v23, "hlsl.shader", 0xBu) )
    {
      v34 = v23;
      v36 = 0;
      v37 = 0;
      v33 = sub_B2D7E0(v23, "hlsl.shader", 0xBu);
      v24 = sub_A72240(&v33);
      v50 = 261;
      v48 = v25;
      v43 = 257;
      v41 = 257;
      v39 = 257;
      v47 = (__int64 *)v24;
      sub_CCA5D0((__int64)&v44, v38, v40, v42, &v47, (__int64)v42);
      v35 = v46;
      v47 = (__int64 *)sub_B2D7E0(v23, "hlsl.numthreads", 0xFu);
      v26 = (char *)sub_A72240((__int64 *)&v47);
      v27 = v42;
      v42[1] = v28;
      v42[0] = v26;
      if ( v28 )
      {
        v48 = 0x300000000LL;
        v47 = (__int64 *)v49;
        sub_C93960(v42, (__int64)&v47, 44, -1, 1, (__int64)v42);
        if ( !sub_C93C90(*v47, v47[1], 0xAu, v40) && v40[0] == LODWORD(v40[0]) )
          v36 = v40[0];
        if ( !sub_C93C90(v47[2], v47[3], 0xAu, v40) && v40[0] == LODWORD(v40[0]) )
          LODWORD(v37) = v40[0];
        if ( !sub_C93C90(v47[4], v47[5], 0xAu, v40) && v40[0] == LODWORD(v40[0]) )
          HIDWORD(v37) = v40[0];
        if ( v47 != (__int64 *)v49 )
          _libc_free((unsigned __int64)v47);
      }
      v17 = *(unsigned int *)(a1 + 64);
      v18 = *(_QWORD *)(a1 + 56);
      v19 = (const __m128i *)&v34;
      v20 = v17 + 1;
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
      {
        v30 = a1 + 56;
        if ( v18 > (unsigned __int64)&v34 || (unsigned __int64)&v34 >= v18 + 24 * v17 )
        {
          sub_C8D5F0(v30, v32, v20, 0x18u, v20, (__int64)v27);
          v18 = *(_QWORD *)(a1 + 56);
          v17 = *(unsigned int *)(a1 + 64);
        }
        else
        {
          v31 = (char *)&v34 - v18;
          sub_C8D5F0(v30, v32, v20, 0x18u, v20, (__int64)v27);
          v18 = *(_QWORD *)(a1 + 56);
          v17 = *(unsigned int *)(a1 + 64);
          v19 = (const __m128i *)&v31[v18];
        }
      }
      v21 = (__m128i *)(v18 + 24 * v17);
      *v21 = _mm_loadu_si128(v19);
      v22 = v44;
      v21[1].m128i_i64[0] = v19[1].m128i_i64[0];
      ++*(_DWORD *)(a1 + 64);
      if ( v22 != &v45 )
        j_j___libc_free_0((unsigned __int64)v22);
    }
  }
  return a1;
}
