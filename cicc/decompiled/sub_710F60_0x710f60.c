// Function: sub_710F60
// Address: 0x710f60
//
unsigned __int64 __fastcall sub_710F60(
        const __m128i *a1,
        __m128i *a2,
        unsigned int a3,
        unsigned int a4,
        int a5,
        int a6,
        int a7,
        int a8,
        _DWORD *a9,
        _DWORD *a10,
        unsigned int *a11,
        _BYTE *a12)
{
  __int64 v12; // r13
  _DWORD *v13; // rbx
  _DWORD *v14; // r15
  __int64 v15; // r14
  __int64 v16; // r9
  int v17; // eax
  _BYTE *v18; // r8
  __int64 v19; // r9
  __int64 i; // rax
  unsigned __int64 result; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rcx
  _QWORD *v26; // r8
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  char v30; // al
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  bool v34; // zf
  __int64 v35; // rax
  __int8 v36; // al
  __int64 v37; // rcx
  __m128i *v38; // rdi
  int v39; // eax
  char v40; // al
  __int64 v41; // rcx
  _DWORD *v42; // rdi
  const __m128i *v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // r13
  _QWORD *v49; // r12
  _QWORD *v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v58; // [rsp+30h] [rbp-70h]
  __int64 v59; // [rsp+38h] [rbp-68h]
  int v60; // [rsp+40h] [rbp-60h] BYREF
  _BOOL4 v61; // [rsp+44h] [rbp-5Ch] BYREF
  __int64 v62; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v63; // [rsp+50h] [rbp-50h] BYREF
  const __m128i *v64; // [rsp+58h] [rbp-48h] BYREF
  __int16 v65[32]; // [rsp+60h] [rbp-40h] BYREF

  v12 = (__int64)a2;
  v13 = a9;
  v14 = a11;
  v15 = a2[8].m128i_i64[0];
  v16 = a1[8].m128i_i64[0];
  *a9 = 0;
  *a11 = 0;
  *a12 = 5;
  v58 = v16;
  v17 = sub_8D2930(v15);
  v18 = a12;
  v19 = v58;
  if ( !v17 )
  {
    result = sub_8D2A90(v15);
    v18 = a12;
    if ( (_DWORD)result )
      goto LABEL_21;
    if ( a7 )
    {
      if ( a6 || (result = sub_710600((__int64)a1), v18 = a12, (_DWORD)result) )
      {
        result = (unsigned __int64)&word_4D04898;
        if ( !word_4D04898 )
          goto LABEL_9;
        result = qword_4D03C50;
        if ( !qword_4D03C50 || *(_BYTE *)(qword_4D03C50 + 16LL) > 2u )
          goto LABEL_9;
      }
LABEL_21:
      *v13 = 1;
      return result;
    }
    result = (unsigned __int64)&dword_4F077C4;
    if ( dword_4F077C4 != 2 )
      goto LABEL_9;
    result = sub_8D2E30(v58);
    v18 = a12;
    if ( !(_DWORD)result )
      goto LABEL_9;
    result = sub_8D2E30(v15);
    v18 = a12;
    if ( !(_DWORD)result )
      goto LABEL_9;
    result = sub_8D5EF0(v58, v15, &v60, &v62);
    v18 = a12;
    if ( !(_DWORD)result )
      goto LABEL_9;
    if ( !a6 )
      goto LABEL_21;
    if ( v60 )
    {
      v24 = sub_8D46C0(v15);
      result = (unsigned __int64)sub_710650(a1, v62, v24, a2, a3, a4, a5, a8, 0, a9, a10, a11);
LABEL_30:
      if ( !*v14 )
      {
        if ( !*v13 )
        {
          v27 = *(_QWORD *)(v12 + 128);
          result = *(unsigned __int8 *)(v27 + 140);
          if ( (_BYTE)result == 12 )
          {
            v28 = *(_QWORD *)(v12 + 128);
            do
            {
              v28 = *(_QWORD *)(v28 + 160);
              result = *(unsigned __int8 *)(v28 + 140);
            }
            while ( (_BYTE)result == 12 );
          }
          if ( (_BYTE)result )
          {
            if ( v27 != v15 )
            {
              result = sub_8D97D0(v27, v15, 0, v25, v26);
              if ( !(_DWORD)result )
                return sub_70C9E0(v12, v15, a5, v22, v23);
            }
          }
        }
        return result;
      }
      *v14 = 0;
      goto LABEL_21;
    }
    v29 = a2[8].m128i_i64[0];
    *a11 = 0;
    v55 = v29;
    v59 = v62;
    sub_8D46C0(v29);
    v30 = *(_BYTE *)(v59 + 96);
    if ( (v30 & 4) != 0 )
    {
      *a11 = 287;
      result = sub_72C970(a2);
      goto LABEL_30;
    }
    if ( (v30 & 2) != 0 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v59 + 112) + 8LL) + 16LL) + 96LL) & 2) != 0 )
    {
      *a11 = 288;
      result = sub_72C970(a2);
      goto LABEL_30;
    }
    v31 = a1[9].m128i_i64[0];
    a1[9].m128i_i64[0] = 0;
    v63 = v31;
    sub_72A510(a1, a2);
    if ( (unsigned int)sub_710600((__int64)a1) )
    {
LABEL_48:
      *(_BYTE *)(v12 + 168) |= 0x28u;
      v34 = v63 == 0;
      *(_QWORD *)(v12 + 128) = v55;
      if ( !v34 )
      {
        v35 = sub_8D46C0(v55);
        sub_6E7880(v35, (_DWORD *)v59, 0, 0, (__int64 *)&v63, a10, &v64);
      }
      result = v63;
      *(_QWORD *)(v12 + 144) = v63;
      goto LABEL_30;
    }
    v64 = (const __m128i *)sub_724DC0(a1, a2, v32, v25, v26, v33);
    v36 = a2[10].m128i_i8[13];
    if ( v36 == 1 )
    {
      v37 = 52;
      v38 = (__m128i *)v64;
      while ( v37 )
      {
        v38->m128i_i32[0] = a2->m128i_i32[0];
        a2 = (__m128i *)((char *)a2 + 4);
        v38 = (__m128i *)((char *)v38 + 4);
        --v37;
      }
    }
    else
    {
      if ( v36 != 6 )
        goto LABEL_72;
      sub_72BAF0(v64, a2[12].m128i_i64[0], unk_4F06A60);
    }
    sub_620DE0(v65, *(_QWORD *)(v59 + 104));
    v39 = sub_620E90((__int64)v64);
    a1 = v64 + 11;
    sub_6215F0((unsigned __int16 *)&v64[11], v65, v39, &v61);
    v40 = *(_BYTE *)(v12 + 173);
    if ( v40 == 1 )
    {
      v41 = 52;
      v42 = (_DWORD *)v12;
      v43 = v64;
      while ( v41 )
      {
        *v42 = v43->m128i_i32[0];
        v43 = (const __m128i *)((char *)v43 + 4);
        ++v42;
        --v41;
      }
LABEL_59:
      sub_724E30(&v64);
      if ( *(_BYTE *)(v12 + 173) == 6 )
      {
        v44 = sub_77F710(v12, 0, 1);
        v45 = *(_QWORD *)(v44 + 16);
        if ( !v45 )
        {
LABEL_67:
          *v14 = 2946;
          result = sub_72C970(v12);
          goto LABEL_30;
        }
        v46 = *(_QWORD *)(v45 + 112);
        v47 = v12;
        v48 = v44;
        v26 = *(_QWORD **)(v46 + 16);
        v49 = *(_QWORD **)(v46 + 8);
        v50 = v26;
        while ( 1 )
        {
          if ( (_QWORD *)*v50 == v49 )
          {
            v12 = v47;
            v13 = a9;
            v14 = a11;
            goto LABEL_67;
          }
          v51 = v49[2];
          v52 = *(_QWORD *)(v51 + 40);
          if ( v55 == v52 )
          {
            v25 = v48;
            v13 = a9;
            v12 = v47;
            v14 = a11;
            goto LABEL_71;
          }
          if ( (unsigned int)sub_8D97D0(v52, v55, 0, v44, v26) )
            break;
          v49 = (_QWORD *)*v49;
        }
        v51 = v49[2];
        v25 = v48;
        v13 = a9;
        v12 = v47;
        v14 = a11;
        if ( !v51 )
          goto LABEL_67;
LABEL_71:
        *(_QWORD *)(v25 + 16) = v51;
      }
      goto LABEL_48;
    }
    if ( v40 == 6 )
    {
      *(_QWORD *)(v12 + 192) = sub_620FA0((__int64)v64, &v61);
      goto LABEL_59;
    }
LABEL_72:
    sub_721090(a1);
  }
  if ( a1[10].m128i_i8[13] == 1 )
    return sub_710080(a1, (__int64)a2, a5, a11, a12);
  for ( i = v15; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(i + 128);
  if ( *(_BYTE *)(v58 + 140) == 12 )
  {
    do
      v19 = *(_QWORD *)(v19 + 160);
    while ( *(_BYTE *)(v19 + 140) == 12 );
  }
  if ( result < *(_QWORD *)(v19 + 128) )
  {
    *a11 = 69;
    *a12 = 8;
  }
LABEL_9:
  if ( !*a9 )
  {
    result = *a11;
    if ( !(_DWORD)result || *v18 != 8 )
    {
      sub_72A510(a1, a2);
      return sub_70C9E0(v12, v15, a5, v22, v23);
    }
  }
  return result;
}
