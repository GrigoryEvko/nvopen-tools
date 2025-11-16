// Function: sub_29DCB40
// Address: 0x29dcb40
//
__int64 __fastcall sub_29DCB40(unsigned int *a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // r13
  unsigned __int8 *v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int8 *v7; // rsi
  __int64 v8; // r12
  __int64 i; // r13
  unsigned __int8 *v10; // rsi
  __int64 result; // rax
  _BYTE *v12; // r15
  _BYTE *v13; // rbx
  __m128i *v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 *v23; // rdx
  __int64 v24; // r10
  __m128i *v25; // r15
  __m128i *v26; // rbx
  __m128i *v27; // rdi
  __int64 (__fastcall *v28)(__int64); // rax
  __int64 v29; // rdi
  int v30; // edx
  int v31; // r11d
  __int64 v32; // [rsp+8h] [rbp-F8h]
  __int64 v33; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v34; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v36; // [rsp+28h] [rbp-D8h]
  __m128i v37; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+40h] [rbp-C0h]
  _BYTE v39[16]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 (__fastcall *v40)(__int64 *); // [rsp+60h] [rbp-A0h]
  __int64 v41; // [rsp+68h] [rbp-98h]
  _BYTE v42[16]; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall *v43)(_QWORD *); // [rsp+80h] [rbp-80h]
  __int64 v44; // [rsp+88h] [rbp-78h]
  __m128i v45; // [rsp+90h] [rbp-70h] BYREF
  __m128i v46; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v47; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v48; // [rsp+B8h] [rbp-48h]
  __int64 v49; // [rsp+C0h] [rbp-40h]
  __int64 v50; // [rsp+C8h] [rbp-38h]

  v1 = *(_QWORD *)a1;
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
  v3 = *(_QWORD *)a1 + 8LL;
  if ( v2 != v3 )
  {
    do
    {
      v4 = (unsigned __int8 *)(v2 - 56);
      if ( !v2 )
        v4 = 0;
      sub_29DC350((__int64)a1, v4);
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v2 != v3 );
    v1 = *(_QWORD *)a1;
  }
  v5 = *(_QWORD *)(v1 + 32);
  v6 = v1 + 24;
  if ( v1 + 24 != v5 )
  {
    do
    {
      v7 = (unsigned __int8 *)(v5 - 56);
      if ( !v5 )
        v7 = 0;
      sub_29DC350((__int64)a1, v7);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v6 != v5 );
    v1 = *(_QWORD *)a1;
  }
  v8 = *(_QWORD *)(v1 + 48);
  for ( i = v1 + 40; i != v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    v10 = (unsigned __int8 *)(v8 - 48);
    if ( !v8 )
      v10 = 0;
    sub_29DC350((__int64)a1, v10);
  }
  result = a1[28];
  if ( (_DWORD)result )
  {
    sub_BA9600(&v45, *(_QWORD *)a1);
    v34 = v47;
    v37 = _mm_loadu_si128(&v45);
    v36 = v48;
    v38 = _mm_loadu_si128(&v46);
    v32 = v49;
    v33 = v50;
    while ( 1 )
    {
      if ( *(_OWORD *)&v37 == __PAIR128__(v36, v34) && v33 == v38.m128i_i64[1] )
      {
        result = v32;
        if ( v32 == v38.m128i_i64[0] )
          return result;
      }
      v12 = v39;
      v41 = 0;
      v13 = v39;
      v14 = &v37;
      v40 = sub_25AC5E0;
      v15 = sub_25AC5C0;
      if ( ((unsigned __int8)sub_25AC5C0 & 1) != 0 )
        goto LABEL_19;
LABEL_20:
      v16 = v15((__int64)v14);
      v17 = v16;
      if ( !v16 )
        break;
LABEL_24:
      v19 = *(_QWORD *)(v16 + 48);
      if ( v19 )
      {
        v20 = *((_QWORD *)a1 + 13);
        v21 = a1[30];
        if ( (_DWORD)v21 )
        {
          v22 = (v21 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v23 = (__int64 *)(v20 + 16LL * v22);
          v24 = *v23;
          if ( v19 == *v23 )
          {
LABEL_27:
            if ( v23 != (__int64 *)(v20 + 16 * v21) )
              sub_B2F990(v17, v23[1], (__int64)v23, v19);
          }
          else
          {
            v30 = 1;
            while ( v24 != -4096 )
            {
              v31 = v30 + 1;
              v22 = (v21 - 1) & (v30 + v22);
              v23 = (__int64 *)(v20 + 16LL * v22);
              v24 = *v23;
              if ( v19 == *v23 )
                goto LABEL_27;
              v30 = v31;
            }
          }
        }
      }
      v25 = (__m128i *)v42;
      v44 = 0;
      v26 = (__m128i *)v42;
      v27 = &v37;
      v43 = sub_25AC590;
      v28 = sub_25AC560;
      if ( ((unsigned __int8)sub_25AC560 & 1) == 0 )
        goto LABEL_31;
LABEL_30:
      v28 = *(__int64 (__fastcall **)(__int64))((char *)v28 + v27->m128i_i64[0] - 1);
LABEL_31:
      while ( !(unsigned __int8)v28((__int64)v27) )
      {
        if ( &v45 == ++v25 )
          goto LABEL_42;
        v29 = v26[1].m128i_i64[1];
        v28 = (__int64 (__fastcall *)(__int64))v26[1].m128i_i64[0];
        v26 = v25;
        v27 = (__m128i *)((char *)&v37 + v29);
        if ( ((unsigned __int8)v28 & 1) != 0 )
          goto LABEL_30;
      }
    }
    while ( 1 )
    {
      v12 += 16;
      if ( v12 == v42 )
LABEL_42:
        BUG();
      v18 = *((_QWORD *)v13 + 3);
      v15 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v13 + 2);
      v13 = v12;
      v14 = (__m128i *)((char *)&v37 + v18);
      if ( ((unsigned __int8)v15 & 1) != 0 )
        break;
      v16 = v15((__int64)v14);
      v17 = v16;
      if ( v16 )
        goto LABEL_24;
    }
LABEL_19:
    v15 = *(__int64 (__fastcall **)(__int64))((char *)v15 + v14->m128i_i64[0] - 1);
    goto LABEL_20;
  }
  return result;
}
