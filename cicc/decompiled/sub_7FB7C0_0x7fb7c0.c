// Function: sub_7FB7C0
// Address: 0x7fb7c0
//
unsigned __int64 __fastcall sub_7FB7C0(
        __int64 a1,
        unsigned int a2,
        __int64 *a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        __m128i *a7)
{
  __m128i *v7; // r14
  int v11; // eax
  int v12; // r15d
  int v13; // eax
  __m128i *i; // r8
  unsigned __int64 result; // rax
  __int64 *v16; // r15
  _QWORD *v17; // rax
  __int64 *v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // rsi
  _QWORD *v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rax
  int v24; // edx
  __m128i *v25; // r12
  __int64 **v26; // rax
  __m128i *v27; // rax
  __int64 v28; // rax
  __m128i *v29; // rax
  int v30; // eax
  _QWORD *v31; // rax
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // r12
  _QWORD *v35; // rax
  __int64 *v36; // rbx
  _QWORD *v37; // rax
  __int64 v38; // rsi
  _QWORD *v39; // rsi
  __m128i *v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __m128i *v42; // [rsp+8h] [rbp-48h]
  __m128i *v43; // [rsp+8h] [rbp-48h]
  __int64 *v45; // [rsp+18h] [rbp-38h] BYREF

  v7 = (__m128i *)a1;
  v45 = a3;
  v11 = sub_8D4070(a1);
  if ( !a5 )
    a5 = 1;
  v12 = v11;
  if ( (unsigned int)sub_8D3410(a1) )
  {
    v41 = sub_8D40F0(a1);
    if ( v12 )
    {
      v7 = (__m128i *)v41;
      a4 = sub_7D78E0(a1);
    }
    else
    {
      v30 = sub_8D23E0(a1);
      if ( a6 || v30 )
      {
        v7 = (__m128i *)v41;
      }
      else
      {
        v7 = (__m128i *)v41;
        a5 *= sub_8D4490(a1);
      }
    }
  }
  v13 = v7[8].m128i_u8[12];
  for ( i = v7; (_BYTE)v13 == 12; v13 = i[8].m128i_u8[12] )
    i = (__m128i *)i[10].m128i_i64[0];
  result = (unsigned int)(v13 - 9);
  if ( (unsigned __int8)result > 2u || (i[11].m128i_i8[3] & 1) == 0 )
  {
    if ( !a6 )
    {
      v43 = i;
      v32 = sub_7E3130((__int64)v7);
      i = v43;
      if ( !v32 )
      {
        v33 = v43[8].m128i_i64[0];
        if ( !a2 && (unsigned __int8)(v43[8].m128i_i8[12] - 9) <= 2u )
          v33 = *(_QWORD *)(v43[10].m128i_i64[1] + 32);
        v34 = v33 * a5;
        if ( a4 )
        {
          v35 = sub_72BA30(byte_4F06A51[0]);
          v36 = (__int64 *)sub_73E130(a4, (__int64)v35);
          v37 = sub_73A8E0(v34, byte_4F06A51[0]);
          v38 = *v36;
          v36[2] = (__int64)v37;
          v39 = sub_73DBF0(0x29u, v38, (__int64)v36);
        }
        else
        {
          v39 = sub_73A8E0(v34, byte_4F06A51[0]);
        }
        return (unsigned __int64)sub_7FA9D0(v45, v39, a7->m128i_i32);
      }
    }
    v16 = v45;
    if ( a4 )
    {
      if ( a5 != 1 )
      {
        v40 = i;
        v17 = sub_72BA30(byte_4F06A51[0]);
        v18 = (__int64 *)sub_73E130(a4, (__int64)v17);
        v19 = sub_73A8E0(a5, byte_4F06A51[0]);
        v20 = *v18;
        v18[2] = (__int64)v19;
        v21 = sub_73DBF0(0x29u, v20, (__int64)v18);
        i = v40;
        a4 = v21;
      }
    }
    else
    {
      if ( a5 == 1 )
      {
        v28 = sub_72D2E0(i);
        v29 = (__m128i *)sub_73E130(v45, v28);
        v24 = 0;
        v45 = (__int64 *)v29;
        v25 = v29;
        v26 = &v45;
        goto LABEL_13;
      }
      v42 = i;
      v31 = sub_73A8E0(a5, byte_4F06A51[0]);
      i = v42;
      a4 = v31;
    }
    v22 = sub_72D2E0(i);
    v23 = sub_73E130(v16, v22);
    v24 = 1;
    v25 = (__m128i *)v23;
    v45 = v23;
    v23[2] = a4;
    v26 = (__int64 **)(v23 + 2);
LABEL_13:
    if ( a6 )
      (*v26)[2] = a6;
    v27 = sub_7FB1A0(v7, a2, v24, a6 == 0, 0);
    return (unsigned __int64)sub_7F88F0((__int64)v27, v25, 0, a7);
  }
  return result;
}
