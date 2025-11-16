// Function: sub_39E0A10
// Address: 0x39e0a10
//
__int64 __fastcall sub_39E0A10(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 *a3,
        size_t a4,
        char *a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v13; // rdi
  void *v14; // rdx
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rax
  char v20; // bl
  size_t v21; // rsi
  unsigned __int64 v22; // rcx
  size_t v23; // r13
  char v24; // r11
  unsigned __int8 *v25; // rdx
  unsigned __int64 v26; // rcx
  char v27; // r13
  unsigned __int8 *v28; // rdx
  size_t v29; // rsi
  size_t v30; // rbx
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int128 v33; // [rsp-10h] [rbp-90h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  size_t v36; // [rsp+18h] [rbp-68h]
  size_t v37; // [rsp+18h] [rbp-68h]
  char v38; // [rsp+26h] [rbp-5Ah]
  unsigned __int8 v39; // [rsp+27h] [rbp-59h]
  char *v41; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v42; // [rsp+30h] [rbp-50h] BYREF
  size_t v43; // [rsp+38h] [rbp-48h]
  _QWORD v44[8]; // [rsp+40h] [rbp-40h] BYREF

  v10 = a1;
  v11 = sub_38BE350(*(_QWORD *)(a1 + 8));
  *((_QWORD *)&v33 + 1) = a6;
  *(_QWORD *)&v33 = a5;
  v39 = sub_39101D0(v11, a1, a2, a3, a4, a7, v33);
  if ( v39 )
  {
    v13 = *(_QWORD *)(a1 + 272);
    v14 = *(void **)(v13 + 24);
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 9u )
    {
      v13 = sub_16E7EE0(v13, "\t.cv_file\t", 0xAu);
    }
    else
    {
      qmemcpy(v14, "\t.cv_file\t", 10);
      *(_QWORD *)(v13 + 24) += 10LL;
    }
    v15 = sub_16E7A90(v13, a2);
    v16 = *(_BYTE **)(v15 + 24);
    if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
    {
      sub_16E7DE0(v15, 32);
    }
    else
    {
      *(_QWORD *)(v15 + 24) = v16 + 1;
      *v16 = 32;
    }
    sub_39E0070(a3, a4, *(_QWORD *)(v10 + 272));
    if ( a7 )
    {
      v17 = *(_QWORD *)(v10 + 272);
      v18 = *(_BYTE **)(v17 + 24);
      if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 16) )
      {
        sub_16E7DE0(v17, 32);
      }
      else
      {
        *(_QWORD *)(v17 + 24) = v18 + 1;
        *v18 = 32;
      }
      v19 = *(_QWORD *)(v10 + 272);
      LOBYTE(v44[0]) = 0;
      v42 = (unsigned __int8 *)v44;
      v35 = v19;
      v43 = 0;
      sub_2240E30((__int64)&v42, 2 * a6);
      if ( a6 )
      {
        v34 = v10;
        v41 = &a5[a6];
        do
        {
          v20 = *a5;
          v21 = v43;
          v22 = 15;
          v23 = v43 + 1;
          v24 = a0123456789abcd[(unsigned __int8)*a5 >> 4];
          v25 = v42;
          if ( v42 != (unsigned __int8 *)v44 )
            v22 = v44[0];
          if ( v23 > v22 )
          {
            v38 = a0123456789abcd[(unsigned __int8)*a5 >> 4];
            v37 = v43;
            sub_2240BB0((unsigned __int64 *)&v42, v43, 0, 0, 1u);
            v25 = v42;
            v24 = v38;
            v21 = v37;
          }
          v25[v21] = v24;
          v26 = 15;
          v43 = v23;
          v27 = a0123456789abcd[v20 & 0xF];
          v42[v21 + 1] = 0;
          v28 = v42;
          v29 = v43;
          if ( v42 != (unsigned __int8 *)v44 )
            v26 = v44[0];
          v30 = v43 + 1;
          if ( v43 + 1 > v26 )
          {
            v36 = v43;
            sub_2240BB0((unsigned __int64 *)&v42, v43, 0, 0, 1u);
            v28 = v42;
            v29 = v36;
          }
          v28[v29] = v27;
          ++a5;
          v43 = v30;
          v42[v29 + 1] = 0;
        }
        while ( v41 != a5 );
        v10 = v34;
      }
      sub_39E0070(v42, v43, v35);
      if ( v42 != (unsigned __int8 *)v44 )
        j_j___libc_free_0((unsigned __int64)v42);
      v31 = *(_QWORD *)(v10 + 272);
      v32 = *(_BYTE **)(v31 + 24);
      if ( (unsigned __int64)v32 >= *(_QWORD *)(v31 + 16) )
      {
        v31 = sub_16E7DE0(v31, 32);
      }
      else
      {
        *(_QWORD *)(v31 + 24) = v32 + 1;
        *v32 = 32;
      }
      sub_16E7A90(v31, a7);
    }
    sub_39E06C0(v10);
  }
  return v39;
}
