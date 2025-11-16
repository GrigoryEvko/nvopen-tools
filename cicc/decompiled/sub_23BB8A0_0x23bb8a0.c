// Function: sub_23BB8A0
// Address: 0x23bb8a0
//
void __fastcall sub_23BB8A0(__m128i **a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  size_t v8; // rdx
  __m128i *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // eax
  unsigned __int64 v13; // rdi
  __int64 v14; // r14
  __int64 v15; // r13
  _QWORD *v16; // r15
  unsigned __int64 v17; // rdi
  __int64 v18; // r12
  size_t v19; // rdx
  __int64 v20; // rdx
  __int64 *v22; // [rsp+20h] [rbp-1E0h]
  int v23; // [rsp+8Ch] [rbp-174h] BYREF
  char *v24; // [rsp+90h] [rbp-170h] BYREF
  size_t v25; // [rsp+98h] [rbp-168h]
  __int64 v26; // [rsp+A0h] [rbp-160h] BYREF
  unsigned __int64 v27[2]; // [rsp+B0h] [rbp-150h] BYREF
  _BYTE v28[16]; // [rsp+C0h] [rbp-140h] BYREF
  void *v29; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v30; // [rsp+D8h] [rbp-128h]
  __int64 v31; // [rsp+E0h] [rbp-120h]
  __int64 v32; // [rsp+E8h] [rbp-118h]
  __int64 v33; // [rsp+F0h] [rbp-110h]
  __int64 v34; // [rsp+F8h] [rbp-108h]
  unsigned __int64 *v35; // [rsp+100h] [rbp-100h]
  __int64 v36; // [rsp+110h] [rbp-F0h] BYREF
  __m128i *v37; // [rsp+118h] [rbp-E8h]
  __m128i *v38; // [rsp+120h] [rbp-E0h]
  _QWORD v39[3]; // [rsp+128h] [rbp-D8h] BYREF
  __int64 v40[4]; // [rsp+140h] [rbp-C0h] BYREF
  char *v41; // [rsp+160h] [rbp-A0h] BYREF
  size_t v42; // [rsp+168h] [rbp-98h]
  __m128i v43; // [rsp+170h] [rbp-90h] BYREF
  __int64 v44; // [rsp+180h] [rbp-80h] BYREF
  void *v45; // [rsp+188h] [rbp-78h] BYREF
  int *v46; // [rsp+190h] [rbp-70h] BYREF
  void **v47; // [rsp+198h] [rbp-68h] BYREF
  _BYTE v48[16]; // [rsp+1A0h] [rbp-60h] BYREF
  unsigned __int64 v49; // [rsp+1B0h] [rbp-50h] BYREF
  unsigned int v50; // [rsp+1B8h] [rbp-48h]
  int v51; // [rsp+1BCh] [rbp-44h]

  v2 = (_BYTE *)sub_2E791E0(a2);
  if ( sub_BC63A0(v2, v3) )
  {
    v4 = (__int64 *)&v29;
    v5 = sub_2E31BC0(*(_QWORD *)(a2 + 328));
    v30 = v6;
    v29 = (void *)v5;
    sub_95CA80((__int64 *)&v41, (__int64)&v29);
    v39[2] = 0x6000000000LL;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39[0] = 0;
    v39[1] = 0;
    sub_2241BD0(v40, (__int64)&v41);
    sub_2240A30((unsigned __int64 *)&v41);
    v7 = *(_QWORD *)(a2 + 328);
    v23 = 0;
    while ( a2 + 320 != v7 )
    {
      v41 = (char *)sub_2E31BC0(v7);
      v42 = v8;
      sub_95CA80((__int64 *)&v24, (__int64)&v41);
      if ( v25 )
      {
        v9 = v37;
        if ( v37 != v38 )
          goto LABEL_5;
      }
      else
      {
        v41 = "{0}";
        LOBYTE(v44) = 1;
        v43.m128i_i64[0] = (__int64)&v47;
        v42 = 3;
        v43.m128i_i64[1] = 1;
        v45 = &unk_4A15FA0;
        v46 = &v23;
        v47 = &v45;
        v34 = 0x100000000LL;
        v27[0] = (unsigned __int64)v28;
        v29 = &unk_49DD210;
        v27[1] = 0;
        v28[0] = 0;
        v30 = 0;
        v31 = 0;
        v32 = 0;
        v33 = 0;
        v35 = v27;
        sub_CB5980((__int64)v4, 0, 0, 0);
        sub_CB6840((__int64)v4, (__int64)&v41);
        if ( v33 != v31 )
          sub_CB5AE0(v4);
        v29 = &unk_49DD210;
        sub_CB5840((__int64)v4);
        sub_23AEBB0((__int64)&v24, (__int64)v27);
        if ( (_BYTE *)v27[0] != v28 )
          j_j___libc_free_0(v27[0]);
        ++v23;
        v9 = v37;
        if ( v37 != v38 )
        {
LABEL_5:
          if ( v9 )
          {
            v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
            sub_23AEDD0(v9->m128i_i64, v24, (__int64)&v24[v25]);
            v9 = v37;
          }
          v37 = v9 + 2;
          goto LABEL_8;
        }
      }
      sub_23BB3E0((unsigned __int64 *)&v36, v9, (__int64)&v24);
LABEL_8:
      v41 = v24;
      v42 = v25;
      v10 = sub_2E31BC0(v7);
      v30 = v11;
      v29 = (void *)v10;
      sub_95CA80(v43.m128i_i64, (__int64)v4);
      v47 = 0;
      v46 = (int *)v48;
      v48[0] = 0;
      sub_23B35A0(&v49, v7);
      v30 = 0;
      v34 = 0x100000000LL;
      v31 = 0;
      v35 = (unsigned __int64 *)&v46;
      v32 = 0;
      v33 = 0;
      v29 = &unk_49DD210;
      sub_CB5980((__int64)v4, 0, 0, 0);
      sub_2E393D0(v7, v4, 0, 1);
      v29 = &unk_49DD210;
      sub_CB5840((__int64)v4);
      v12 = sub_C92610();
      sub_23BB630((__int64)v39, v41, v42, v12, &v43);
      if ( v51 )
      {
        v13 = v49;
        if ( v50 )
        {
          v22 = v4;
          v14 = 8LL * v50;
          v15 = 0;
          do
          {
            v16 = *(_QWORD **)(v13 + v15);
            if ( v16 != (_QWORD *)-8LL && v16 )
            {
              v17 = v16[1];
              v18 = *v16 + 41LL;
              if ( (_QWORD *)v17 != v16 + 3 )
                j_j___libc_free_0(v17);
              sub_C7D6A0((__int64)v16, v18, 8);
              v13 = v49;
            }
            v15 += 8;
          }
          while ( v14 != v15 );
          v4 = v22;
        }
      }
      else
      {
        v13 = v49;
      }
      _libc_free(v13);
      if ( v46 != (int *)v48 )
        j_j___libc_free_0((unsigned __int64)v46);
      if ( (__int64 *)v43.m128i_i64[0] != &v44 )
        j_j___libc_free_0(v43.m128i_u64[0]);
      if ( v24 != (char *)&v26 )
        j_j___libc_free_0((unsigned __int64)v24);
      v7 = *(_QWORD *)(v7 + 8);
    }
    v41 = (char *)sub_2E791E0(a2);
    v42 = v19;
    sub_23BB830(a1, &v41);
    v41 = (char *)sub_2E791E0(a2);
    v42 = v20;
    sub_23B7BC0((__int64 **)&v43, &v36, v20);
    sub_23B7FB0((__int64)(a1 + 3), (__int64)&v41);
    sub_23B6480((__int64)&v43);
    sub_23B6480((__int64)&v36);
  }
}
