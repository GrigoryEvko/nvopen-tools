// Function: sub_D58140
// Address: 0xd58140
//
__int64 __fastcall sub_D58140(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  bool v6; // bl
  unsigned int v7; // r15d
  unsigned __int8 (__fastcall *v9)(__int64, __int64, __int64, const void *, const void *); // r10
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __m128i *v13; // rdx
  __int64 v14; // r12
  __m128i si128; // xmm0
  __int64 v16; // rax
  size_t v17; // rdx
  void *v18; // rdi
  unsigned __int8 *v19; // rsi
  unsigned __int64 v20; // rax
  const char *v21; // rax
  size_t v22; // rdx
  _BYTE *v23; // rdi
  unsigned __int8 *v24; // rsi
  _BYTE *v25; // rax
  size_t v26; // r13
  char *v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // r13
  __int64 *v30; // rdi
  __int64 v31; // rsi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // r15
  _QWORD *v35; // r12
  unsigned __int64 v36; // r13
  unsigned __int64 v37; // rdx
  _QWORD *v38; // r14
  __int64 v39; // r15
  _QWORD *v40; // rbx
  _QWORD *v41; // rax
  __int64 v42; // r8
  _QWORD *v43; // r12
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // r14
  _QWORD *v46; // rcx
  unsigned __int64 v47; // rsi
  size_t v48; // rdx
  int v49; // eax
  _QWORD *v50; // r14
  __int64 *v51; // rax
  __int64 *v52; // rbx
  char v53; // al
  unsigned __int64 v54; // r9
  __int64 *v55; // r14
  __int64 ***v56; // rax
  __int64 *v57; // rdx
  size_t v58; // r15
  _QWORD *v59; // rsi
  unsigned __int64 v60; // rdi
  _QWORD *v61; // rcx
  unsigned __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  void *v66; // rax
  __int64 *v67; // rcx
  __int64 v68; // rax
  void *v69; // rdx
  __int64 v70; // [rsp+0h] [rbp-80h]
  unsigned __int8 (__fastcall *v71)(__int64, __int64, __int64, const void *, const void *); // [rsp+8h] [rbp-78h]
  __int64 v72; // [rsp+8h] [rbp-78h]
  _QWORD *v73; // [rsp+10h] [rbp-70h]
  const void *v74; // [rsp+18h] [rbp-68h]
  __int64 v75; // [rsp+18h] [rbp-68h]
  const void *v76; // [rsp+20h] [rbp-60h]
  _QWORD *v77; // [rsp+20h] [rbp-60h]
  unsigned __int64 v78; // [rsp+20h] [rbp-60h]
  unsigned __int64 v79; // [rsp+20h] [rbp-60h]
  size_t v81; // [rsp+28h] [rbp-58h]
  const void *v82[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v83[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = **(_QWORD **)(a2 + 32);
  v3 = *(_QWORD *)(v2 + 72);
  if ( v3 )
  {
    v4 = sub_B2BE50(*(_QWORD *)(v2 + 72));
    v5 = sub_B6F960(v4);
    if ( !byte_4F876E0 && (unsigned int)sub_2207590(&byte_4F876E0) )
    {
      qword_4F87708 = 1;
      qword_4F87710 = 0;
      v29 = (_QWORD *)qword_4F82348[8];
      qword_4F87718 = 0;
      v30 = &qword_4F87730 - 2;
      qword_4F87700 = (__int64)&qword_4F87730;
      v73 = (_QWORD *)qword_4F82348[9];
      dword_4F87720 = 1065353216;
      qword_4F87728 = 0;
      v31 = (__int64)(qword_4F82348[9] - qword_4F82348[8]) >> 5;
      qword_4F87730 = 0;
      v32 = sub_222D860(&qword_4F87730 - 2, v31);
      v34 = v32;
      if ( v32 > qword_4F87708 )
      {
        if ( v32 == 1 )
        {
          qword_4F87730 = 0;
          v67 = &qword_4F87730;
        }
        else
        {
          if ( v32 > 0xFFFFFFFFFFFFFFFLL )
LABEL_82:
            sub_4261EA(v30, v31, v33);
          v66 = (void *)sub_22077B0(8 * v32);
          v67 = (__int64 *)memset(v66, 0, 8 * v34);
        }
        qword_4F87700 = (__int64)v67;
        qword_4F87708 = v34;
      }
      if ( v73 != v29 )
      {
        v70 = v3;
        v72 = v5;
        v35 = v29;
        do
        {
          v36 = sub_22076E0(*v35, v35[1], 3339675911LL);
          v37 = v36 % qword_4F87708;
          v38 = *(_QWORD **)(qword_4F87700 + 8 * (v36 % qword_4F87708));
          v39 = 8 * (v36 % qword_4F87708);
          if ( !v38 )
            goto LABEL_46;
          v40 = (_QWORD *)*v38;
          v41 = v35;
          v42 = 8 * v37;
          v43 = *(_QWORD **)(qword_4F87700 + 8 * v37);
          v44 = v36 % qword_4F87708;
          v45 = qword_4F87708;
          v46 = v41;
          v47 = v40[5];
          while ( 1 )
          {
            if ( v36 == v47 )
            {
              v48 = v46[1];
              if ( v48 == v40[2] )
              {
                if ( !v48
                  || (v75 = v42,
                      v77 = v46,
                      v49 = memcmp((const void *)*v46, (const void *)v40[1], v48),
                      v46 = v77,
                      v42 = v75,
                      !v49) )
                {
                  v50 = v43;
                  v39 = v42;
                  v35 = v46;
                  if ( !*v50 )
                    goto LABEL_46;
                  goto LABEL_38;
                }
              }
            }
            if ( !*v40 )
              break;
            v47 = *(_QWORD *)(*v40 + 40LL);
            v43 = v40;
            if ( v44 != v47 % v45 )
              break;
            v40 = (_QWORD *)*v40;
          }
          v39 = v42;
          v35 = v46;
LABEL_46:
          v51 = (__int64 *)sub_22077B0(48);
          v52 = v51;
          if ( v51 )
            *v51 = 0;
          v51[1] = (__int64)(v51 + 3);
          sub_D575B0(v51 + 1, (_BYTE *)*v35, *v35 + v35[1]);
          v31 = qword_4F87708;
          v30 = (__int64 *)&dword_4F87720;
          v53 = sub_222DA10(&dword_4F87720, qword_4F87708, qword_4F87718, 1);
          v54 = v33;
          if ( !v53 )
          {
            v55 = (__int64 *)qword_4F87700;
            goto LABEL_50;
          }
          if ( v33 == 1 )
          {
            qword_4F87730 = 0;
            v55 = &qword_4F87730;
            goto LABEL_56;
          }
          if ( v33 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_82;
          v58 = 8 * v33;
          v78 = v33;
          v55 = (__int64 *)sub_22077B0(8 * v33);
          memset(v55, 0, v58);
          v54 = v78;
LABEL_56:
          v59 = (_QWORD *)qword_4F87710;
          qword_4F87710 = 0;
          if ( v59 )
          {
            v60 = 0;
            do
            {
              v61 = v59;
              v59 = (_QWORD *)*v59;
              v62 = v61[5] % v54;
              v63 = &v55[v62];
              if ( *v63 )
              {
                *v61 = *(_QWORD *)*v63;
                *(_QWORD *)*v63 = v61;
              }
              else
              {
                *v61 = qword_4F87710;
                qword_4F87710 = (__int64)v61;
                *v63 = (__int64)&qword_4F87710;
                if ( *v61 )
                  v55[v60] = (__int64)v61;
                v60 = v62;
              }
            }
            while ( v59 );
          }
          if ( (__int64 *)qword_4F87700 != &qword_4F87730 )
          {
            v79 = v54;
            j_j___libc_free_0(qword_4F87700, 8 * qword_4F87708);
            v54 = v79;
          }
          qword_4F87708 = v54;
          qword_4F87700 = (__int64)v55;
          v39 = 8 * (v36 % v54);
LABEL_50:
          v56 = (__int64 ***)((char *)v55 + v39);
          v52[5] = v36;
          v57 = *(__int64 **)((char *)v55 + v39);
          if ( v57 )
          {
            *v52 = *v57;
            **v56 = v52;
          }
          else
          {
            v64 = qword_4F87710;
            qword_4F87710 = (__int64)v52;
            *v52 = v64;
            if ( v64 )
            {
              v55[*(_QWORD *)(v64 + 40) % (unsigned __int64)qword_4F87708] = (__int64)v52;
              v56 = (__int64 ***)(v39 + qword_4F87700);
            }
            *v56 = (__int64 **)&qword_4F87710;
          }
          ++qword_4F87718;
LABEL_38:
          v35 += 4;
        }
        while ( v73 != v35 );
        v5 = v72;
        v3 = v70;
      }
      __cxa_atexit((void (*)(void *))sub_8565C0, &qword_4F87700, &qword_4A427C0);
      sub_2207640(&byte_4F876E0);
    }
    v6 = 0;
    if ( qword_4F87718 )
    {
      v27 = (char *)sub_BD5D20(v3);
      v82[0] = v83;
      sub_D57500((__int64 *)v82, v27, (__int64)&v27[v28]);
      v6 = sub_BB97F0(&qword_4F87700, v82) == 0;
      if ( v82[0] != v83 )
        j_j___libc_free_0(v82[0], v83[0] + 1LL);
    }
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 24LL))(v5);
    if ( !(_BYTE)v7 )
      return sub_B2D610(v3, 48);
    v9 = *(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, const void *, const void *))(*(_QWORD *)v5 + 16LL);
    v82[0] = v83;
    v71 = v9;
    sub_D57500((__int64 *)v82, "loop", (__int64)"");
    v74 = v82[0];
    v76 = v82[1];
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    if ( v71(v5, v10, v11, v74, v76) )
    {
      if ( v82[0] != v83 )
        j_j___libc_free_0(v82[0], v83[0] + 1LL);
      return sub_B2D610(v3, 48);
    }
    if ( v82[0] != v83 )
      j_j___libc_free_0(v82[0], v83[0] + 1LL);
    if ( v6 )
    {
      v12 = sub_CB72A0();
      v13 = (__m128i *)v12[4];
      v14 = (__int64)v12;
      if ( v12[3] - (_QWORD)v13 <= 0x17u )
      {
        v14 = sub_CB6200((__int64)v12, "BISECT: Skip bisecting '", 0x18u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F560C0);
        v13[1].m128i_i64[0] = 0x2720676E69746365LL;
        *v13 = si128;
        v12[4] += 24LL;
      }
      v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      v18 = *(void **)(v14 + 32);
      v19 = (unsigned __int8 *)v16;
      v20 = *(_QWORD *)(v14 + 24) - (_QWORD)v18;
      if ( v20 < v17 )
      {
        v65 = sub_CB6200(v14, v19, v17);
        v18 = *(void **)(v65 + 32);
        v14 = v65;
        v20 = *(_QWORD *)(v65 + 24) - (_QWORD)v18;
      }
      else if ( v17 )
      {
        v81 = v17;
        memcpy(v18, v19, v17);
        v68 = *(_QWORD *)(v14 + 24);
        v69 = (void *)(*(_QWORD *)(v14 + 32) + v81);
        *(_QWORD *)(v14 + 32) = v69;
        v18 = v69;
        v20 = v68 - (_QWORD)v69;
      }
      if ( v20 <= 0xD )
      {
        v14 = sub_CB6200(v14, "' on function ", 0xEu);
      }
      else
      {
        qmemcpy(v18, "' on function ", 14);
        *(_QWORD *)(v14 + 32) += 14LL;
      }
      v21 = sub_BD5D20(v3);
      v23 = *(_BYTE **)(v14 + 32);
      v24 = (unsigned __int8 *)v21;
      v25 = *(_BYTE **)(v14 + 24);
      v26 = v22;
      if ( v22 > v25 - v23 )
      {
        v14 = sub_CB6200(v14, v24, v22);
        v25 = *(_BYTE **)(v14 + 24);
        v23 = *(_BYTE **)(v14 + 32);
      }
      else if ( v22 )
      {
        memcpy(v23, v24, v22);
        v25 = *(_BYTE **)(v14 + 24);
        v23 = (_BYTE *)(v26 + *(_QWORD *)(v14 + 32));
        *(_QWORD *)(v14 + 32) = v23;
      }
      if ( v25 == v23 )
      {
        sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v23 = 10;
        ++*(_QWORD *)(v14 + 32);
      }
      return sub_B2D610(v3, 48);
    }
  }
  else
  {
    return 0;
  }
  return v7;
}
