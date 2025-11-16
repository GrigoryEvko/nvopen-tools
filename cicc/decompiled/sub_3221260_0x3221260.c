// Function: sub_3221260
// Address: 0x3221260
//
__int64 __fastcall sub_3221260(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // rax
  unsigned __int8 v7; // cl
  unsigned __int8 v8; // dl
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  _BYTE *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r10
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdx
  unsigned __int8 v21; // cl
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 result; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rdx
  bool v36; // cf
  unsigned __int8 v37; // al
  unsigned int v38; // r13d
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // r8
  __m128i *v43; // rax
  size_t v44; // rcx
  __m128i *v45; // r10
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdi
  __m128i *v48; // rax
  __m128i *v49; // rcx
  __m128i *v50; // rdx
  __int64 v51; // [rsp+0h] [rbp-E0h]
  __int64 v52; // [rsp+8h] [rbp-D8h]
  __int64 v53; // [rsp+10h] [rbp-D0h]
  __int64 v54; // [rsp+18h] [rbp-C8h]
  const char *v55; // [rsp+20h] [rbp-C0h]
  __int64 v56; // [rsp+28h] [rbp-B8h]
  __m128i *v57; // [rsp+30h] [rbp-B0h]
  __int64 v58; // [rsp+38h] [rbp-A8h]
  __m128i v59; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v60; // [rsp+50h] [rbp-90h] BYREF
  __int64 v61; // [rsp+58h] [rbp-88h]
  _BYTE v62[16]; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v63; // [rsp+70h] [rbp-70h] BYREF
  size_t v64; // [rsp+78h] [rbp-68h]
  __m128i v65; // [rsp+80h] [rbp-60h] BYREF
  char *v66; // [rsp+90h] [rbp-50h] BYREF
  size_t v67; // [rsp+98h] [rbp-48h]
  _QWORD v68[8]; // [rsp+A0h] [rbp-40h] BYREF

  v3 = a3 + 8;
  v56 = a2 - 16;
  v6 = a2;
  if ( *(_BYTE *)a2 != 16 )
  {
    v7 = *(_BYTE *)(a2 - 16);
    if ( (v7 & 2) != 0 )
    {
      v6 = **(_QWORD **)(a2 - 32);
      if ( !v6 )
      {
        v54 = 0;
        v55 = byte_3F871B3;
        goto LABEL_9;
      }
    }
    else
    {
      v6 = *(_QWORD *)(v56 - 8LL * ((v7 >> 2) & 0xF));
      if ( !v6 )
      {
        v54 = 0;
        v55 = byte_3F871B3;
        goto LABEL_46;
      }
    }
  }
  v8 = *(_BYTE *)(v6 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(__int64 **)(v6 - 32);
  else
    v9 = (__int64 *)(v6 - 16 - 8LL * ((v8 >> 2) & 0xF));
  v10 = *v9;
  if ( *v9 )
    v10 = sub_B91420(v10);
  else
    v11 = 0;
  v7 = *(_BYTE *)(a2 - 16);
  v55 = (const char *)v10;
  v54 = v11;
  if ( (v7 & 2) != 0 )
  {
LABEL_9:
    v12 = *(_BYTE **)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( !v12 )
    {
      v51 = 0;
      v15 = 0;
      v16 = 0;
      goto LABEL_11;
    }
    goto LABEL_10;
  }
LABEL_46:
  v12 = *(_BYTE **)(v56 - 8LL * ((v7 >> 2) & 0xF) + 8);
  if ( !v12 )
  {
    v51 = 0;
    v15 = 0;
    v16 = 0;
    goto LABEL_48;
  }
LABEL_10:
  v13 = sub_B91420((__int64)v12);
  v7 = *(_BYTE *)(a2 - 16);
  v51 = v14;
  v12 = (_BYTE *)v13;
  v15 = v13;
  v16 = v14;
  if ( (v7 & 2) == 0 )
  {
LABEL_48:
    v17 = v56 - 8LL * ((v7 >> 2) & 0xF);
    goto LABEL_12;
  }
LABEL_11:
  v17 = *(_QWORD *)(a2 - 32);
LABEL_12:
  v18 = *(_QWORD *)(v17 + 16);
  if ( !v18
    || (v52 = v16, v53 = v15, v19 = (_BYTE *)sub_B91420(v18), v15 = v53, v16 = v52, !v20)
    || *(_BYTE *)(a1 + 3768) )
  {
    sub_324AD70(a3, v3, 37, v15, v16);
    goto LABEL_16;
  }
  if ( v19 )
  {
    v66 = (char *)v68;
    sub_3219430((__int64 *)&v66, v19, (__int64)&v19[v20]);
    if ( v12 )
      goto LABEL_63;
LABEL_84:
    v62[0] = 0;
    v60 = v62;
    v61 = 0;
    goto LABEL_64;
  }
  v67 = 0;
  v66 = (char *)v68;
  LOBYTE(v68[0]) = 0;
  if ( !v12 )
    goto LABEL_84;
LABEL_63:
  v60 = v62;
  sub_3219430((__int64 *)&v60, v12, (__int64)&v12[v51]);
  if ( v61 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
LABEL_64:
  v43 = (__m128i *)sub_2241490((unsigned __int64 *)&v60, " ", 1u);
  v63 = &v65;
  if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
  {
    v65 = _mm_loadu_si128(v43 + 1);
  }
  else
  {
    v63 = (__m128i *)v43->m128i_i64[0];
    v65.m128i_i64[0] = v43[1].m128i_i64[0];
  }
  v44 = v43->m128i_u64[1];
  v43[1].m128i_i8[0] = 0;
  v64 = v44;
  v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
  v45 = v63;
  v43->m128i_i64[1] = 0;
  v46 = 15;
  v47 = 15;
  if ( v45 != &v65 )
    v47 = v65.m128i_i64[0];
  if ( v64 + v67 > v47 )
  {
    if ( v66 != (char *)v68 )
      v46 = v68[0];
    if ( v64 + v67 <= v46 )
    {
      v48 = (__m128i *)sub_2241130((unsigned __int64 *)&v66, 0, 0, v45, v64);
      v57 = &v59;
      v49 = (__m128i *)v48->m128i_i64[0];
      v50 = v48 + 1;
      if ( (__m128i *)v48->m128i_i64[0] != &v48[1] )
        goto LABEL_73;
LABEL_92:
      v59 = _mm_loadu_si128(v48 + 1);
      goto LABEL_74;
    }
  }
  v48 = (__m128i *)sub_2241490((unsigned __int64 *)&v63, v66, v67);
  v57 = &v59;
  v49 = (__m128i *)v48->m128i_i64[0];
  v50 = v48 + 1;
  if ( (__m128i *)v48->m128i_i64[0] == &v48[1] )
    goto LABEL_92;
LABEL_73:
  v57 = v49;
  v59.m128i_i64[0] = v48[1].m128i_i64[0];
LABEL_74:
  v58 = v48->m128i_i64[1];
  v48->m128i_i64[0] = (__int64)v50;
  v48->m128i_i64[1] = 0;
  v48[1].m128i_i8[0] = 0;
  if ( v63 != &v65 )
    j_j___libc_free_0((unsigned __int64)v63);
  if ( v60 != (_QWORD *)v62 )
    j_j___libc_free_0((unsigned __int64)v60);
  if ( v66 != (char *)v68 )
    j_j___libc_free_0((unsigned __int64)v66);
  sub_324AD70(a3, v3, 37, v57, v58);
  if ( v57 != &v59 )
    j_j___libc_free_0((unsigned __int64)v57);
LABEL_16:
  LODWORD(v66) = 65541;
  sub_3249A20(a3, a3 + 16, 19, v66, *(unsigned int *)(a2 + 16));
  sub_324AD70(a3, v3, 3, v55, v54);
  v21 = *(_BYTE *)(a2 - 16);
  if ( (v21 & 2) == 0 )
  {
    v22 = *(_QWORD *)(v56 - 8LL * ((v21 >> 2) & 0xF) + 72);
    if ( !v22 )
      goto LABEL_50;
    goto LABEL_18;
  }
  v22 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 72LL);
  if ( v22 )
  {
LABEL_18:
    v23 = sub_B91420(v22);
    if ( v24 )
      sub_324AD70(a3, v3, 15874, v23, v24);
    v21 = *(_BYTE *)(a2 - 16);
    if ( (v21 & 2) != 0 )
      goto LABEL_21;
LABEL_50:
    result = v56 - 8LL * ((v21 >> 2) & 0xF);
    goto LABEL_22;
  }
LABEL_21:
  result = *(_QWORD *)(a2 - 32);
LABEL_22:
  v26 = *(_QWORD *)(result + 80);
  if ( v26 )
  {
    result = sub_B91420(v26);
    if ( v27 )
      result = sub_324AD70(a3, v3, 16367, result, v27);
  }
  if ( !*(_BYTE *)(a1 + 3769) )
  {
    if ( *(_BYTE *)(a1 + 3770) )
      sub_324ACD0(a3);
    sub_3735CB0(a3);
    if ( *(_QWORD *)(a1 + 3072) )
      sub_324AD70(a3, v3, 27, *(_QWORD *)(a1 + 3064), *(_QWORD *)(a1 + 3072));
    result = sub_321F6B0(a1, a3, v3);
  }
  if ( *(_BYTE *)(a1 + 3768) )
  {
    if ( *(_BYTE *)(a2 + 40) )
      sub_3249FA0(a3, v3, 16353);
    result = *(unsigned __int8 *)(a2 - 16);
    if ( (result & 2) != 0 )
    {
      v28 = *(_QWORD *)(a2 - 32);
    }
    else
    {
      result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
      v28 = v56 - result;
    }
    v29 = *(_QWORD *)(v28 + 16);
    if ( v29 )
    {
      result = sub_B91420(v29);
      if ( v30 )
        result = sub_324AD70(a3, v3, 16354, result, v30);
    }
    v31 = *(unsigned int *)(a2 + 20);
    if ( (_DWORD)v31 )
    {
      LODWORD(v66) = 65547;
      result = sub_3249A20(a3, a3 + 16, 16357, v66, v31);
    }
  }
  v32 = *(_QWORD *)(a2 + 24);
  if ( v32 )
  {
    LODWORD(v66) = 65543;
    sub_3249A20(a3, a3 + 16, 8497, v66, v32);
    result = *(unsigned __int8 *)(a2 - 16);
    if ( (result & 2) != 0 )
    {
      v33 = *(_QWORD *)(a2 - 32);
    }
    else
    {
      result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
      v33 = v56 - result;
    }
    v34 = *(_QWORD *)(v33 + 24);
    if ( v34 )
    {
      result = sub_B91420(v34);
      if ( v35 )
      {
        v36 = (unsigned __int16)sub_3220AA0(a1) < 5u;
        v37 = *(_BYTE *)(a2 - 16);
        v38 = v36 ? 8496 : 118;
        if ( (v37 & 2) != 0 )
          v39 = *(_QWORD *)(a2 - 32);
        else
          v39 = v56 - 8LL * ((v37 >> 2) & 0xF);
        v40 = *(_QWORD *)(v39 + 24);
        if ( v40 )
        {
          v40 = sub_B91420(*(_QWORD *)(v39 + 24));
          v42 = v41;
        }
        else
        {
          v42 = 0;
        }
        return sub_324AD70(a3, v3, v38, v40, v42);
      }
    }
  }
  return result;
}
