// Function: sub_2D21D20
// Address: 0x2d21d20
//
__int64 __fastcall sub_2D21D20(unsigned int **a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // ebx
  char v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  char v13; // cl
  __int64 v14; // r12
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // rdi
  __m128i si128; // xmm0
  __int64 v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rdi
  __m128i v22; // xmm0
  __int64 v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rbx
  __m128i v26; // xmm0
  unsigned __int8 *v27; // rax
  size_t v28; // rdx
  char *v29; // rcx
  void *v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v34; // [rsp+0h] [rbp-E0h]
  unsigned int v35; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v36; // [rsp+Eh] [rbp-D2h]
  unsigned __int8 v37; // [rsp+Fh] [rbp-D1h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  char v40; // [rsp+20h] [rbp-C0h]
  __int64 v41; // [rsp+28h] [rbp-B8h]
  size_t v42; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v43[2]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v44[2]; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int8 *v45[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v46; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v47; // [rsp+70h] [rbp-70h] BYREF
  __int64 v48; // [rsp+78h] [rbp-68h]
  __int64 v49; // [rsp+80h] [rbp-60h]
  __int64 v50; // [rsp+88h] [rbp-58h]
  __int64 v51; // [rsp+90h] [rbp-50h]
  __int64 v52; // [rsp+98h] [rbp-48h]
  unsigned __int64 *v53; // [rsp+A0h] [rbp-40h]

  v34 = a2 + 24;
  v35 = **a1;
  v39 = *(_QWORD *)(a2 + 32);
  if ( v39 == a2 + 24 )
  {
    return 0;
  }
  else
  {
    v36 = 0;
    v2 = a2 + 312;
    do
    {
      v3 = 0;
      if ( v39 )
        v3 = v39 - 56;
      v38 = v3;
      v4 = v3;
      v37 = sub_CE9220(v3);
      if ( v37 )
      {
        if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v38, a2, v5, v6);
          v7 = *(_QWORD *)(v38 + 96);
          if ( (*(_BYTE *)(v38 + 2) & 1) != 0 )
            sub_B2C6D0(v38, a2, v31, v32);
          v8 = *(_QWORD *)(v38 + 96);
        }
        else
        {
          v7 = *(_QWORD *)(v38 + 96);
          v8 = v7;
        }
        v41 = v8 + 40LL * *(_QWORD *)(v38 + 104);
        if ( v41 != v7 )
        {
          v9 = 0;
          do
          {
            v14 = *(_QWORD *)(v7 + 8);
            if ( (unsigned __int8)sub_B2D680(v7) )
            {
              a2 = sub_B2BD20(v7);
              v10 = sub_AE5020(v2, a2);
              v11 = sub_9208B0(v2, a2);
              v13 = v10;
            }
            else
            {
              a2 = v14;
              v40 = sub_AE5020(v2, v14);
              v11 = sub_9208B0(v2, v14);
              v13 = v40;
            }
            v48 = v12;
            v7 += 40;
            v47 = ((1LL << v13) + ((unsigned __int64)(v11 + 7) >> 3) - 1) >> v13 << v13;
            v9 += sub_CA1930(&v47);
          }
          while ( v7 != v41 );
          if ( v35 < v9 )
          {
            v52 = 0x100000000LL;
            v53 = v43;
            v43[0] = (unsigned __int64)v44;
            v47 = (unsigned __int64)&unk_49DD210;
            v43[1] = 0;
            LOBYTE(v44[0]) = 0;
            v48 = 0;
            v49 = 0;
            v50 = 0;
            v51 = 0;
            sub_CB5980((__int64)&v47, 0, 0, 0);
            sub_2C75640((__int64 *)v45, v38);
            v15 = sub_CB6200((__int64)&v47, v45[0], (size_t)v45[1]);
            v16 = *(__m128i **)(v15 + 32);
            v17 = v15;
            if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0x2Bu )
            {
              v17 = sub_CB6200(v15, ": Error: Formal parameter space overflowed (", 0x2Cu);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_42E1EE0);
              qmemcpy(&v16[2], "overflowed (", 12);
              *v16 = si128;
              v16[1] = _mm_load_si128((const __m128i *)&xmmword_42E1EF0);
              *(_QWORD *)(v15 + 32) += 44LL;
            }
            v19 = sub_CB59D0(v17, v9);
            v20 = *(__m128i **)(v19 + 32);
            v21 = v19;
            if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 0x14u )
            {
              v21 = sub_CB6200(v19, " bytes required, max ", 0x15u);
            }
            else
            {
              v22 = _mm_load_si128((const __m128i *)&xmmword_42E1F00);
              v20[1].m128i_i32[0] = 2019650848;
              v20[1].m128i_i8[4] = 32;
              *v20 = v22;
              *(_QWORD *)(v19 + 32) += 21LL;
            }
            v23 = sub_CB59D0(v21, v35);
            v24 = *(__m128i **)(v23 + 32);
            v25 = v23;
            if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 0x1Bu )
            {
              v25 = sub_CB6200(v23, " bytes allowed) in function ", 0x1Cu);
            }
            else
            {
              v26 = _mm_load_si128((const __m128i *)&xmmword_42E1F10);
              qmemcpy(&v24[1], "in function ", 12);
              *v24 = v26;
              *(_QWORD *)(v23 + 32) += 28LL;
            }
            v27 = (unsigned __int8 *)sub_BD5D20(v38);
            v30 = *(void **)(v25 + 32);
            if ( v28 > *(_QWORD *)(v25 + 24) - (_QWORD)v30 )
            {
              sub_CB6200(v25, v27, v28);
            }
            else if ( v28 )
            {
              v42 = v28;
              memcpy(v30, v27, v28);
              v28 = v42;
              *(_QWORD *)(v25 + 32) += v42;
            }
            if ( (__int64 *)v45[0] != &v46 )
              j_j___libc_free_0((unsigned __int64)v45[0]);
            a2 = 1;
            sub_CEB590(v53, 1, v28, v29);
            v47 = (unsigned __int64)&unk_49DD210;
            sub_CB5840((__int64)&v47);
            if ( (_QWORD *)v43[0] != v44 )
            {
              a2 = v44[0] + 1LL;
              j_j___libc_free_0(v43[0]);
            }
            v36 = v37;
          }
        }
      }
      v39 = *(_QWORD *)(v39 + 8);
    }
    while ( v34 != v39 );
  }
  return v36;
}
