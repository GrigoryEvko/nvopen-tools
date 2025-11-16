// Function: sub_309F310
// Address: 0x309f310
//
void __fastcall sub_309F310(int a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned __int8 v7; // bl
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r12
  __int64 v14; // r13
  size_t v15; // rdx
  int v16; // eax
  _QWORD *v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  _WORD *v21; // rdx
  void *v22; // rax
  _QWORD *v23; // rax
  _WORD *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  _WORD *v27; // rdx
  void *v28; // rax
  _QWORD *v29; // rax
  _WORD *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdi
  _BYTE *v33; // rax
  unsigned __int64 v34; // rdi
  __int64 v35; // r12
  _QWORD *v36; // rax
  void *v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rdi
  _BYTE *v40; // rax
  _QWORD *v41; // rax
  void *v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rdi
  _BYTE *v45; // rax
  unsigned __int8 v48; // [rsp+33h] [rbp-11Dh]
  int v49; // [rsp+34h] [rbp-11Ch]
  __int64 v50; // [rsp+38h] [rbp-118h]
  unsigned int v53; // [rsp+58h] [rbp-F8h]
  unsigned int v54; // [rsp+5Ch] [rbp-F4h]
  __m128i s2; // [rsp+60h] [rbp-F0h] BYREF
  _BYTE v56[16]; // [rsp+70h] [rbp-E0h] BYREF
  __m128i s1; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE v58[16]; // [rsp+90h] [rbp-C0h] BYREF
  _QWORD v59[8]; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD v60[14]; // [rsp+E0h] [rbp-70h] BYREF

  v7 = a1;
  if ( (_BYTE)qword_502E088 || a2 )
  {
    v8 = a4;
    v50 = a6;
    v9 = *(_QWORD *)(a3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v9 + 16);
    v10 = *(_DWORD *)(v9 + 8);
    v11 = *(_QWORD *)(a5 + 8);
    v53 = v10 >> 8;
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
      v11 = **(_QWORD **)(v11 + 16);
    v12 = *(_DWORD *)(v11 + 8);
    v59[6] = &s2;
    v54 = v12 >> 8;
    v48 = BYTE1(a1) & 1;
    v49 = a1 >> 9;
    s2.m128i_i64[0] = (__int64)v56;
    s1.m128i_i64[0] = (__int64)v58;
    v59[5] = 0x100000000LL;
    v59[0] = &unk_49DD210;
    s2.m128i_i64[1] = 0;
    v56[0] = 0;
    s1.m128i_i64[1] = 0;
    v58[0] = 0;
    memset(&v59[1], 0, 32);
    sub_CB5980((__int64)v59, 0, 0, 0);
    v60[5] = 0x100000000LL;
    v60[6] = &s1;
    v60[0] = &unk_49DD210;
    memset(&v60[1], 0, 32);
    sub_CB5980((__int64)v60, 0, 0, 0);
    sub_A5BF40((unsigned __int8 *)a3, (__int64)v59, 0, a7);
    sub_A5BF40((unsigned __int8 *)a5, (__int64)v60, 0, a7);
    v60[0] = &unk_49DD210;
    sub_CB5840((__int64)v60);
    v59[0] = &unk_49DD210;
    sub_CB5840((__int64)v59);
    v13 = s1.m128i_i64[1];
    v14 = s2.m128i_i64[1];
    v15 = s2.m128i_u64[1];
    if ( s1.m128i_i64[1] <= (unsigned __int64)s2.m128i_i64[1] )
      v15 = s1.m128i_u64[1];
    if ( v15 && (v16 = memcmp((const void *)s1.m128i_i64[0], (const void *)s2.m128i_i64[0], v15)) != 0 )
    {
      if ( v16 >= 0 )
        goto LABEL_12;
    }
    else
    {
      v35 = v13 - v14;
      if ( v35 > 0x7FFFFFFF || v35 >= (__int64)0xFFFFFFFF80000000LL && (int)v35 >= 0 )
        goto LABEL_12;
    }
    sub_22415E0(&s2, &s1);
    if ( (a1 & 0x100) != 0 )
    {
      if ( v49 == -4194304 )
      {
        v49 = -4194304;
        v8 = a6;
        v54 = v53;
        v53 = v12 >> 8;
        v50 = a4;
LABEL_12:
        v17 = sub_CB72A0();
        v18 = (_WORD *)v17[4];
        v19 = (__int64)v17;
        if ( v17[3] - (_QWORD)v18 <= 1u )
        {
          v19 = sub_CB6200((__int64)v17, (unsigned __int8 *)"  ", 2u);
        }
        else
        {
          *v18 = 8224;
          v17[4] += 2LL;
        }
        v20 = sub_CF5E90(v19, v7 | (v48 << 8) | (unsigned int)(v49 << 9));
        v21 = *(_WORD **)(v20 + 32);
        if ( *(_QWORD *)(v20 + 24) - (_QWORD)v21 <= 1u )
        {
          sub_CB6200(v20, (unsigned __int8 *)":\t", 2u);
        }
        else
        {
          *v21 = 2362;
          *(_QWORD *)(v20 + 32) += 2LL;
        }
        v22 = sub_CB72A0();
        sub_A587F0(v8, (__int64)v22, 0, 1);
        if ( v53 )
        {
          v36 = sub_CB72A0();
          v37 = (void *)v36[4];
          v38 = (__int64)v36;
          if ( v36[3] - (_QWORD)v37 <= 0xAu )
          {
            v38 = sub_CB6200((__int64)v36, (unsigned __int8 *)" addrspace(", 0xBu);
          }
          else
          {
            qmemcpy(v37, " addrspace(", 11);
            v36[4] += 11LL;
          }
          v39 = sub_CB59D0(v38, v53);
          v40 = *(_BYTE **)(v39 + 32);
          if ( *(_BYTE **)(v39 + 24) == v40 )
          {
            sub_CB6200(v39, (unsigned __int8 *)")", 1u);
          }
          else
          {
            *v40 = 41;
            ++*(_QWORD *)(v39 + 32);
          }
        }
        v23 = sub_CB72A0();
        v24 = (_WORD *)v23[4];
        v25 = (__int64)v23;
        if ( v23[3] - (_QWORD)v24 <= 1u )
        {
          v25 = sub_CB6200((__int64)v23, (unsigned __int8 *)"* ", 2u);
        }
        else
        {
          *v24 = 8234;
          v23[4] += 2LL;
        }
        v26 = sub_CB6200(v25, (unsigned __int8 *)s2.m128i_i64[0], s2.m128i_u64[1]);
        v27 = *(_WORD **)(v26 + 32);
        if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 1u )
        {
          sub_CB6200(v26, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v27 = 8236;
          *(_QWORD *)(v26 + 32) += 2LL;
        }
        v28 = sub_CB72A0();
        sub_A587F0(v50, (__int64)v28, 0, 1);
        if ( v54 )
        {
          v41 = sub_CB72A0();
          v42 = (void *)v41[4];
          v43 = (__int64)v41;
          if ( v41[3] - (_QWORD)v42 <= 0xAu )
          {
            v43 = sub_CB6200((__int64)v41, (unsigned __int8 *)" addrspace(", 0xBu);
          }
          else
          {
            qmemcpy(v42, " addrspace(", 11);
            v41[4] += 11LL;
          }
          v44 = sub_CB59D0(v43, v54);
          v45 = *(_BYTE **)(v44 + 32);
          if ( *(_BYTE **)(v44 + 24) == v45 )
          {
            sub_CB6200(v44, (unsigned __int8 *)")", 1u);
          }
          else
          {
            *v45 = 41;
            ++*(_QWORD *)(v44 + 32);
          }
        }
        v29 = sub_CB72A0();
        v30 = (_WORD *)v29[4];
        v31 = (__int64)v29;
        if ( v29[3] - (_QWORD)v30 <= 1u )
        {
          v31 = sub_CB6200((__int64)v29, (unsigned __int8 *)"* ", 2u);
        }
        else
        {
          *v30 = 8234;
          v29[4] += 2LL;
        }
        v32 = sub_CB6200(v31, (unsigned __int8 *)s1.m128i_i64[0], s1.m128i_u64[1]);
        v33 = *(_BYTE **)(v32 + 32);
        if ( *(_BYTE **)(v32 + 24) == v33 )
        {
          sub_CB6200(v32, (unsigned __int8 *)"\n", 1u);
          v34 = s1.m128i_i64[0];
          if ( (_BYTE *)s1.m128i_i64[0] == v58 )
          {
LABEL_27:
            if ( (_BYTE *)s2.m128i_i64[0] != v56 )
              j_j___libc_free_0(s2.m128i_u64[0]);
            return;
          }
        }
        else
        {
          *v33 = 10;
          ++*(_QWORD *)(v32 + 32);
          v34 = s1.m128i_i64[0];
          if ( (_BYTE *)s1.m128i_i64[0] == v58 )
            goto LABEL_27;
        }
        j_j___libc_free_0(v34);
        goto LABEL_27;
      }
      v48 = BYTE1(a1) & 1;
      v49 = (-512 * v49) >> 9;
    }
    v8 = a6;
    v54 = v53;
    v53 = v12 >> 8;
    v50 = a4;
    goto LABEL_12;
  }
}
