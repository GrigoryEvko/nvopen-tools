// Function: sub_37425F0
// Address: 0x37425f0
//
void __fastcall sub_37425F0(__m128i *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 *v6; // rdx
  __int64 *v7; // r12
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  __int64 *v16; // r8
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // rbx
  unsigned __int8 *v23; // r15
  __int64 v24; // rsi
  void (__fastcall *v25)(__m128i *, unsigned __int8 *, __int64, __int64, __m128i *); // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rcx
  _QWORD *v30; // rax
  __int64 v31; // rdi
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // rsi
  __int64 (__fastcall *v35)(__int64, __int64, _QWORD *, __int64, unsigned __int8 **); // rax
  __int64 v36; // r14
  _QWORD *v37; // rdx
  __int64 v38; // [rsp+0h] [rbp-B0h]
  __int64 *v39; // [rsp+8h] [rbp-A8h]
  __int64 *v40; // [rsp+8h] [rbp-A8h]
  __int64 *v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+10h] [rbp-A0h]
  __int64 v43; // [rsp+10h] [rbp-A0h]
  _QWORD *v44; // [rsp+18h] [rbp-98h]
  void (__fastcall *v45)(__m128i *, unsigned __int8 *, __int64, __int64, __m128i *); // [rsp+18h] [rbp-98h]
  __int64 (__fastcall *v46)(__int64, __int64, _QWORD *, __int64, unsigned __int8 **); // [rsp+18h] [rbp-98h]
  unsigned __int8 *v47; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int8 *v48; // [rsp+30h] [rbp-80h] BYREF
  __int64 v49; // [rsp+38h] [rbp-78h]
  __int64 v50; // [rsp+40h] [rbp-70h]
  __m128i v51; // [rsp+50h] [rbp-60h] BYREF
  __int64 v52; // [rsp+60h] [rbp-50h]
  __int64 v53; // [rsp+68h] [rbp-48h]

  if ( sub_B44020(a2) )
  {
    v4 = a1[5].m128i_i64[0];
    v51 = 0u;
    v52 = 0;
    if ( v4 )
    {
      sub_B91220((__int64)a1[5].m128i_i64, v4);
      a1[5] = v51;
      a1[6].m128i_i64[0] = v52;
    }
    else
    {
      a1[5].m128i_i64[1] = 0;
      a1[6].m128i_i64[0] = 0;
    }
    v5 = *(_QWORD *)(a2 + 64);
    if ( v5 )
    {
      v38 = sub_B14240(v5);
      v7 = v6;
      if ( (__int64 *)v38 != v6 )
      {
        do
        {
          v21 = *v7;
          sub_3741110((__int64)a1);
          sub_3741080((__int64)a1);
          v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_BYTE *)(v22 + 32) == 1 )
          {
            v8 = *(unsigned __int8 **)(v22 + 24);
            v9 = *(_QWORD *)(a1[7].m128i_i64[1] + 8);
            v47 = v8;
            v10 = v9 - 720;
            if ( v8 )
            {
              sub_B96E90((__int64)&v47, (__int64)v8, 1);
              v48 = v47;
              if ( v47 )
              {
                sub_B976B0((__int64)&v47, v47, (__int64)&v48);
                v11 = a1[2].m128i_i64[1];
                v47 = 0;
                v49 = 0;
                v12 = *(_QWORD *)(v11 + 744);
                v13 = *(__int64 **)(v11 + 752);
                v50 = 0;
                v14 = *(_QWORD **)(v12 + 32);
                v42 = v12;
                v51.m128i_i64[0] = (__int64)v48;
                v44 = v14;
                if ( v48 )
                {
                  v39 = v13;
                  sub_B96E90((__int64)&v51, (__int64)v48, 1);
                  v13 = v39;
                }
                goto LABEL_12;
              }
            }
            else
            {
              v48 = 0;
            }
            v28 = a1[2].m128i_i64[1];
            v49 = 0;
            v50 = 0;
            v29 = *(_QWORD *)(v28 + 744);
            v13 = *(__int64 **)(v28 + 752);
            v30 = *(_QWORD **)(v29 + 32);
            v42 = v29;
            v51.m128i_i64[0] = 0;
            v44 = v30;
LABEL_12:
            v40 = v13;
            v15 = sub_2E7B380(v44, v10, (unsigned __int8 **)&v51, 0);
            v16 = v40;
            v17 = (__int64)v15;
            if ( v51.m128i_i64[0] )
            {
              sub_B91220((__int64)&v51, v51.m128i_i64[0]);
              v16 = v40;
            }
            v41 = v16;
            sub_2E31040((__int64 *)(v42 + 40), v17);
            v18 = *v41;
            v19 = *(_QWORD *)v17 & 7LL;
            *(_QWORD *)(v17 + 8) = v41;
            v18 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v17 = v18 | v19;
            *(_QWORD *)(v18 + 8) = v17;
            *v41 = v17 | *v41 & 7;
            if ( v49 )
              sub_2E882B0(v17, (__int64)v44, v49);
            if ( v50 )
              sub_2E88680(v17, (__int64)v44, v50);
            v20 = sub_B11FB0(v22 + 40);
            v51.m128i_i64[0] = 14;
            v52 = 0;
            v53 = v20;
            sub_2E8EAD0(v17, (__int64)v44, &v51);
            if ( v48 )
              sub_B91220((__int64)&v48, (__int64)v48);
            if ( v47 )
              sub_B91220((__int64)&v47, (__int64)v47);
            goto LABEL_22;
          }
          v23 = 0;
          if ( **(_BYTE **)(v22 + 40) != 4 )
            v23 = (unsigned __int8 *)sub_B12A50(v22, 0);
          if ( (unsigned __int8)(*(_BYTE *)(v22 + 64) - 1) > 1u )
          {
            v31 = a1[2].m128i_i64[1];
            if ( *(_BYTE *)(v31 + 1020) )
            {
              v32 = *(_QWORD **)(v31 + 1000);
              v33 = &v32[*(unsigned int *)(v31 + 1012)];
              if ( v32 != v33 )
              {
                while ( v22 != *v32 )
                {
                  if ( v33 == ++v32 )
                    goto LABEL_38;
                }
                goto LABEL_22;
              }
            }
            else if ( sub_C8CA60(v31 + 992, v22) )
            {
              goto LABEL_22;
            }
LABEL_38:
            v34 = *(_QWORD *)(v22 + 24);
            v35 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, unsigned __int8 **))(a1->m128i_i64[0] + 136);
            v51.m128i_i64[0] = v34;
            v46 = v35;
            if ( v34 )
              sub_B96E90((__int64)&v51, v34, 1);
            v36 = sub_B12000(v22 + 72);
            v37 = (_QWORD *)sub_B11F60(v22 + 80);
            if ( v46 == sub_37425D0 )
            {
              if ( v23 && (unsigned int)*v23 - 12 > 1 )
                sub_3742340((__int64)a1, (__int64)v23, v37, v36, (unsigned __int8 **)&v51);
            }
            else
            {
              v46((__int64)a1, (__int64)v23, v37, v36, (unsigned __int8 **)&v51);
            }
            v27 = v51.m128i_i64[0];
            if ( !v51.m128i_i64[0] )
              goto LABEL_22;
          }
          else
          {
            v24 = *(_QWORD *)(v22 + 24);
            v25 = *(void (__fastcall **)(__m128i *, unsigned __int8 *, __int64, __int64, __m128i *))(a1->m128i_i64[0] + 128);
            v51.m128i_i64[0] = v24;
            v45 = v25;
            if ( v24 )
              sub_B96E90((__int64)&v51, v24, 1);
            v43 = sub_B12000(v22 + 72);
            v26 = sub_B11F60(v22 + 80);
            v45(a1, v23, v26, v43, &v51);
            v27 = v51.m128i_i64[0];
            if ( !v51.m128i_i64[0] )
              goto LABEL_22;
          }
          sub_B91220((__int64)&v51, v27);
LABEL_22:
          v7 = (__int64 *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
        }
        while ( v7 != (__int64 *)v38 );
      }
    }
  }
}
