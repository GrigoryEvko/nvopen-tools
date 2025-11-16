// Function: sub_23BC0E0
// Address: 0x23bc0e0
//
void __fastcall sub_23BC0E0(__m128i **a1, __int64 a2)
{
  char *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 *v5; // r13
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdx
  char *v11; // rsi
  __m128i *v12; // rdi
  __int64 v13; // rdx
  char *v14; // rsi
  int v15; // eax
  unsigned __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v18; // r15
  _QWORD *v19; // r14
  unsigned __int64 v20; // rdi
  __int64 v21; // r13
  size_t v22; // rdx
  const char *v23; // rax
  __int64 v24; // rdx
  int v25; // eax
  __int64 *v28; // [rsp+20h] [rbp-1E0h]
  __int64 i; // [rsp+78h] [rbp-188h]
  int v30; // [rsp+8Ch] [rbp-174h] BYREF
  char *v31; // [rsp+90h] [rbp-170h] BYREF
  size_t v32; // [rsp+98h] [rbp-168h]
  _BYTE v33[16]; // [rsp+A0h] [rbp-160h] BYREF
  unsigned __int64 v34[2]; // [rsp+B0h] [rbp-150h] BYREF
  _BYTE v35[16]; // [rsp+C0h] [rbp-140h] BYREF
  const char *v36; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v37; // [rsp+D8h] [rbp-128h]
  __int64 v38; // [rsp+E0h] [rbp-120h]
  __int64 v39; // [rsp+E8h] [rbp-118h]
  __int64 v40; // [rsp+F0h] [rbp-110h]
  __int64 v41; // [rsp+F8h] [rbp-108h]
  unsigned __int64 *v42; // [rsp+100h] [rbp-100h]
  __int64 v43; // [rsp+110h] [rbp-F0h] BYREF
  __m128i *v44; // [rsp+118h] [rbp-E8h]
  __m128i *v45; // [rsp+120h] [rbp-E0h]
  _QWORD v46[3]; // [rsp+128h] [rbp-D8h] BYREF
  __int64 v47[4]; // [rsp+140h] [rbp-C0h] BYREF
  const char *v48; // [rsp+160h] [rbp-A0h] BYREF
  size_t v49; // [rsp+168h] [rbp-98h]
  __m128i v50; // [rsp+170h] [rbp-90h] BYREF
  __int64 v51; // [rsp+180h] [rbp-80h] BYREF
  void *v52; // [rsp+188h] [rbp-78h] BYREF
  int *v53; // [rsp+190h] [rbp-70h] BYREF
  void **v54; // [rsp+198h] [rbp-68h] BYREF
  _BYTE v55[16]; // [rsp+1A0h] [rbp-60h] BYREF
  unsigned __int64 v56; // [rsp+1B0h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+1B8h] [rbp-48h]
  int v58; // [rsp+1BCh] [rbp-44h]

  if ( !sub_B2FC80(a2) )
  {
    v2 = (char *)sub_BD5D20(a2);
    if ( sub_BC63A0(v2, v3) )
    {
      v4 = *(_QWORD *)(a2 + 80);
      v5 = (__int64 *)&v36;
      if ( v4 )
        v4 -= 24;
      v6 = sub_BD5D20(v4);
      v37 = v7;
      v36 = v6;
      sub_95CA80((__int64 *)&v48, (__int64)&v36);
      v46[2] = 0x6000000000LL;
      v43 = 0;
      v44 = 0;
      v45 = 0;
      v46[0] = 0;
      v46[1] = 0;
      sub_2241BD0(v47, (__int64)&v48);
      sub_2240A30((unsigned __int64 *)&v48);
      v8 = *(_QWORD *)(a2 + 80);
      v30 = 0;
      for ( i = v8; a2 + 72 != i; i = *(_QWORD *)(i + 8) )
      {
        v9 = i - 24;
        if ( !i )
          v9 = 0;
        v11 = (char *)sub_BD5D20(v9);
        v31 = v33;
        if ( v11 )
        {
          sub_23AE760((__int64 *)&v31, v11, (__int64)&v11[v10]);
          if ( v32 )
          {
            v12 = v44;
            if ( v44 != v45 )
              goto LABEL_12;
            goto LABEL_43;
          }
        }
        else
        {
          v32 = 0;
          v33[0] = 0;
        }
        v48 = "{0}";
        v50.m128i_i64[0] = (__int64)&v54;
        v49 = 3;
        LOBYTE(v51) = 1;
        v52 = &unk_4A15FA0;
        v53 = &v30;
        v54 = &v52;
        v41 = 0x100000000LL;
        v50.m128i_i64[1] = 1;
        v36 = (const char *)&unk_49DD210;
        v34[0] = (unsigned __int64)v35;
        v34[1] = 0;
        v35[0] = 0;
        v37 = 0;
        v38 = 0;
        v39 = 0;
        v40 = 0;
        v42 = v34;
        sub_CB5980((__int64)v5, 0, 0, 0);
        sub_CB6840((__int64)v5, (__int64)&v48);
        if ( v40 != v38 )
          sub_CB5AE0(v5);
        v36 = (const char *)&unk_49DD210;
        sub_CB5840((__int64)v5);
        sub_23AEBB0((__int64)&v31, (__int64)v34);
        if ( (_BYTE *)v34[0] != v35 )
          j_j___libc_free_0(v34[0]);
        ++v30;
        v12 = v44;
        if ( v44 != v45 )
        {
LABEL_12:
          if ( v12 )
          {
            v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
            sub_23AEDD0(v12->m128i_i64, v31, (__int64)&v31[v32]);
            v12 = v44;
          }
          v44 = v12 + 2;
          goto LABEL_15;
        }
LABEL_43:
        sub_23BB3E0((unsigned __int64 *)&v43, v12, (__int64)&v31);
LABEL_15:
        v48 = v31;
        v49 = v32;
        v14 = (char *)sub_BD5D20(v9);
        v50.m128i_i64[0] = (__int64)&v51;
        if ( v14 )
        {
          sub_23AE760(v50.m128i_i64, v14, (__int64)&v14[v13]);
        }
        else
        {
          v50.m128i_i64[1] = 0;
          LOBYTE(v51) = 0;
        }
        v54 = 0;
        v55[0] = 0;
        v53 = (int *)v55;
        sub_23B30B0(&v56, v9);
        v37 = 0;
        v41 = 0x100000000LL;
        v38 = 0;
        v42 = (unsigned __int64 *)&v53;
        v39 = 0;
        v40 = 0;
        v36 = (const char *)&unk_49DD210;
        sub_CB5980((__int64)v5, 0, 0, 0);
        sub_A68DD0(v9, (__int64)v5, 0, 1, 1);
        v36 = (const char *)&unk_49DD210;
        sub_CB5840((__int64)v5);
        v15 = sub_C92610();
        sub_23BB630((__int64)v46, v48, v49, v15, &v50);
        if ( v58 )
        {
          v16 = v56;
          if ( v57 )
          {
            v28 = v5;
            v17 = 8LL * v57;
            v18 = 0;
            do
            {
              v19 = *(_QWORD **)(v16 + v18);
              if ( v19 != (_QWORD *)-8LL && v19 )
              {
                v20 = v19[1];
                v21 = *v19 + 41LL;
                if ( (_QWORD *)v20 != v19 + 3 )
                  j_j___libc_free_0(v20);
                sub_C7D6A0((__int64)v19, v21, 8);
                v16 = v56;
              }
              v18 += 8;
            }
            while ( v17 != v18 );
            v5 = v28;
          }
        }
        else
        {
          v16 = v56;
        }
        _libc_free(v16);
        if ( v53 != (int *)v55 )
          j_j___libc_free_0((unsigned __int64)v53);
        if ( (__int64 *)v50.m128i_i64[0] != &v51 )
          j_j___libc_free_0(v50.m128i_u64[0]);
        if ( v31 != v33 )
          j_j___libc_free_0((unsigned __int64)v31);
      }
      v48 = sub_BD5D20(a2);
      v49 = v22;
      sub_23BB830(a1, &v48);
      v23 = sub_BD5D20(a2);
      v49 = v24;
      v48 = v23;
      sub_23B7BC0((__int64 **)&v50, &v43, v24);
      v25 = sub_C92610();
      sub_23BBF00((__int64)(a1 + 3), v48, v49, v25, &v50);
      sub_23B6480((__int64)&v50);
      sub_23B6480((__int64)&v43);
    }
  }
}
