// Function: sub_3222C60
// Address: 0x3222c60
//
__int64 __fastcall sub_3222C60(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r12
  char v4; // si
  _BYTE *v5; // rdi
  unsigned __int8 v6; // al
  _BYTE *v7; // rcx
  bool v8; // dl
  _QWORD *v9; // r8
  __int64 v10; // rdi
  char v11; // r13
  __m128i v12; // rax
  __int64 v13; // rdx
  char v14; // cl
  _BYTE *v15; // rdx
  unsigned __int8 v16; // al
  _BYTE *v17; // rsi
  unsigned __int8 v18; // al
  __int64 *v19; // rdx
  char *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  unsigned __int8 v24; // dl
  __int64 v25; // rax
  char *v26; // rsi
  __int64 v27; // rdx
  bool v28; // zf
  __m128i v29; // xmm1
  char v30; // r12
  __m128i v31; // xmm0
  _QWORD *v32; // rdi
  size_t v33; // r9
  __int64 v34; // rsi
  __int64 v35; // rdx
  _QWORD *v36; // rdi
  size_t v37; // r8
  __int64 v38; // rsi
  __int64 v39; // rdx
  char v40; // al
  __m128i v41; // xmm2
  __int64 v42; // rax
  __m128i v43; // xmm3
  __int64 *v45; // rcx
  _QWORD *v46; // rsi
  size_t v47; // rdx
  size_t v48; // rdx
  __int64 v49; // [rsp-120h] [rbp-120h]
  __m128i v50; // [rsp-118h] [rbp-118h] BYREF
  char v51; // [rsp-108h] [rbp-108h]
  __m128i v52; // [rsp-D8h] [rbp-D8h] BYREF
  char v53; // [rsp-C8h] [rbp-C8h]
  __m128i v54; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v55; // [rsp-A8h] [rbp-A8h]
  __m128i v56; // [rsp-78h] [rbp-78h] BYREF
  __int64 v57; // [rsp-68h] [rbp-68h]
  _QWORD *v58; // [rsp-58h] [rbp-58h] BYREF
  size_t v59; // [rsp-50h] [rbp-50h]
  _QWORD v60[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)(a1 + 3769) )
  {
    v3 = *(_BYTE **)(a2 + 80);
    v4 = *v3;
    v5 = v3;
    if ( *v3 == 16
      || ((v6 = *(v3 - 16), v7 = v3 - 16, v8 = (v6 & 2) != 0, (v6 & 2) == 0)
        ? (v9 = &v7[-8 * ((v6 >> 2) & 0xF)])
        : (v9 = (_QWORD *)*((_QWORD *)v3 - 4)),
          (v5 = (_BYTE *)*v9) != 0) )
    {
      v10 = *((_QWORD *)v5 + 5);
      v11 = 0;
      if ( v10 )
      {
        v12.m128i_i64[0] = sub_B91420(v10);
        v4 = *v3;
        v11 = 1;
        v54 = v12;
      }
      v13 = (__int64)v3;
      if ( v4 == 16 )
      {
LABEL_9:
        sub_3222AF0(&v50, a1, v13);
        v14 = *v3;
        v15 = v3;
        if ( *v3 != 16 )
        {
          v16 = *(v3 - 16);
          v17 = v3 - 16;
          if ( (v16 & 2) != 0 )
          {
            v15 = (_BYTE *)**((_QWORD **)v3 - 4);
            if ( !v15 )
            {
              v49 = 0;
              v20 = (char *)byte_3F871B3;
LABEL_41:
              v46 = (_QWORD *)*((_QWORD *)v3 - 4);
              goto LABEL_42;
            }
          }
          else
          {
            v15 = *(_BYTE **)&v17[-8 * ((v16 >> 2) & 0xF)];
            if ( !v15 )
            {
              v49 = 0;
              v20 = (char *)byte_3F871B3;
              goto LABEL_49;
            }
          }
        }
        v18 = *(v15 - 16);
        if ( (v18 & 2) != 0 )
          v19 = (__int64 *)*((_QWORD *)v15 - 4);
        else
          v19 = (__int64 *)&v15[-8 * ((v18 >> 2) & 0xF) - 16];
        v20 = (char *)*v19;
        if ( *v19 )
        {
          v21 = sub_B91420(*v19);
          v14 = *v3;
          v49 = v22;
          v20 = (char *)v21;
        }
        else
        {
          v49 = 0;
        }
        v23 = v3;
        if ( v14 == 16 )
          goto LABEL_17;
        v16 = *(v3 - 16);
        v17 = v3 - 16;
        if ( (v16 & 2) != 0 )
          goto LABEL_41;
LABEL_49:
        v46 = &v17[-8 * ((v16 >> 2) & 0xF)];
LABEL_42:
        v23 = (_BYTE *)*v46;
        if ( !*v46 )
        {
          v27 = 0;
          v26 = (char *)byte_3F871B3;
          goto LABEL_21;
        }
LABEL_17:
        v24 = *(v23 - 16);
        if ( (v24 & 2) != 0 )
          v25 = *((_QWORD *)v23 - 4);
        else
          v25 = (__int64)&v23[-8 * ((v24 >> 2) & 0xF) - 16];
        v26 = *(char **)(v25 + 8);
        if ( v26 )
          v26 = (char *)sub_B91420(*(_QWORD *)(v25 + 8));
        else
          v27 = 0;
LABEL_21:
        v28 = *(_QWORD *)(a1 + 4712) == 0;
        LOBYTE(v55) = v11;
        v29 = _mm_loadu_si128(&v50);
        v30 = v51;
        v31 = _mm_loadu_si128(&v54);
        if ( !v28 )
          return a1 + 4272;
        v57 = v55;
        v53 = v51;
        v58 = v60;
        v52 = v29;
        v56 = v31;
        sub_3219430((__int64 *)&v58, v26, (__int64)&v26[v27]);
        v32 = *(_QWORD **)(a1 + 4672);
        if ( v58 == v60 )
        {
          v48 = v59;
          if ( v59 )
          {
            if ( v59 == 1 )
              *(_BYTE *)v32 = v60[0];
            else
              memcpy(v32, v60, v59);
            v48 = v59;
            v32 = *(_QWORD **)(a1 + 4672);
          }
          *(_QWORD *)(a1 + 4680) = v48;
          *((_BYTE *)v32 + v48) = 0;
          v32 = v58;
        }
        else
        {
          v33 = v59;
          v34 = v60[0];
          if ( v32 == (_QWORD *)(a1 + 4688) )
          {
            *(_QWORD *)(a1 + 4672) = v58;
            *(_QWORD *)(a1 + 4680) = v33;
            *(_QWORD *)(a1 + 4688) = v34;
          }
          else
          {
            v35 = *(_QWORD *)(a1 + 4688);
            *(_QWORD *)(a1 + 4672) = v58;
            *(_QWORD *)(a1 + 4680) = v33;
            *(_QWORD *)(a1 + 4688) = v34;
            if ( v32 )
            {
              v58 = v32;
              v60[0] = v35;
              goto LABEL_26;
            }
          }
          v58 = v60;
          v32 = v60;
        }
LABEL_26:
        v59 = 0;
        *(_BYTE *)v32 = 0;
        if ( v58 != v60 )
          j_j___libc_free_0((unsigned __int64)v58);
        v58 = v60;
        sub_3219430((__int64 *)&v58, v20, (__int64)&v20[v49]);
        v36 = *(_QWORD **)(a1 + 4704);
        if ( v58 == v60 )
        {
          v47 = v59;
          if ( v59 )
          {
            if ( v59 == 1 )
              *(_BYTE *)v36 = v60[0];
            else
              memcpy(v36, v60, v59);
            v47 = v59;
            v36 = *(_QWORD **)(a1 + 4704);
          }
          *(_QWORD *)(a1 + 4712) = v47;
          *((_BYTE *)v36 + v47) = 0;
          v36 = v58;
          goto LABEL_32;
        }
        v37 = v59;
        v38 = v60[0];
        if ( v36 == (_QWORD *)(a1 + 4720) )
        {
          *(_QWORD *)(a1 + 4704) = v58;
          *(_QWORD *)(a1 + 4712) = v37;
          *(_QWORD *)(a1 + 4720) = v38;
        }
        else
        {
          v39 = *(_QWORD *)(a1 + 4720);
          *(_QWORD *)(a1 + 4704) = v58;
          *(_QWORD *)(a1 + 4712) = v37;
          *(_QWORD *)(a1 + 4720) = v38;
          if ( v36 )
          {
            v58 = v36;
            v60[0] = v39;
LABEL_32:
            v59 = 0;
            *(_BYTE *)v36 = 0;
            if ( v58 != v60 )
              j_j___libc_free_0((unsigned __int64)v58);
            v40 = v53;
            *(_BYTE *)(a1 + 4785) &= v30;
            *(_BYTE *)(a1 + 4786) |= v30;
            v41 = _mm_loadu_si128(&v52);
            *(_BYTE *)(a1 + 4756) = v40;
            v42 = v57;
            *(_BYTE *)(a1 + 4784) |= v11;
            v43 = _mm_loadu_si128(&v56);
            *(_DWORD *)(a1 + 4736) = 0;
            *(_QWORD *)(a1 + 4776) = v42;
            *(__m128i *)(a1 + 4740) = v41;
            *(__m128i *)(a1 + 4760) = v43;
            return a1 + 4272;
          }
        }
        v58 = v60;
        v36 = v60;
        goto LABEL_32;
      }
      v6 = *(v3 - 16);
      v7 = v3 - 16;
      v8 = (v6 & 2) != 0;
    }
    else
    {
      v11 = 0;
    }
    if ( v8 )
      v45 = (__int64 *)*((_QWORD *)v3 - 4);
    else
      v45 = (__int64 *)&v7[-8 * ((v6 >> 2) & 0xF)];
    v13 = *v45;
    goto LABEL_9;
  }
  return 0;
}
