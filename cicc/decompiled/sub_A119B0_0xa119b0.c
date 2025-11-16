// Function: sub_A119B0
// Address: 0xa119b0
//
__int64 *__fastcall sub_A119B0(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, int a5)
{
  unsigned int v7; // r12d
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // ecx
  int *v12; // r14
  int v13; // edi
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  _BYTE *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdi
  _QWORD *v21; // rax
  const char *v22; // rax
  int v24; // edx
  __int64 v27; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 v29; // [rsp+20h] [rbp-D0h]
  _QWORD *v30; // [rsp+20h] [rbp-D0h]
  __int64 v32[4]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v33[4]; // [rsp+50h] [rbp-A0h] BYREF
  const char *v34; // [rsp+70h] [rbp-80h] BYREF
  __int64 v35; // [rsp+78h] [rbp-78h]
  __int64 v36; // [rsp+80h] [rbp-70h]
  __int64 v37; // [rsp+88h] [rbp-68h]
  __int64 v38; // [rsp+90h] [rbp-60h]
  unsigned __int64 v39; // [rsp+98h] [rbp-58h]
  __int64 v40; // [rsp+A0h] [rbp-50h]
  __int64 v41; // [rsp+A8h] [rbp-48h]
  __int64 v42; // [rsp+B0h] [rbp-40h]
  __int64 v43; // [rsp+B8h] [rbp-38h]

  if ( !a5 )
  {
LABEL_11:
    *a1 = 1;
    return a1;
  }
  v7 = 0;
  while ( 1 )
  {
    v8 = a2[67].m128i_i64[0];
    v9 = *(_QWORD *)(a4 + 8LL * v7);
    v10 = a2[68].m128i_u32[0];
    if ( !(_DWORD)v10 )
      break;
    v11 = (v10 - 1) & (37 * v9);
    v12 = (int *)(v8 + 8LL * v11);
    v13 = *v12;
    if ( (_DWORD)v9 != *v12 )
    {
      v24 = 1;
      while ( v13 != -1 )
      {
        v11 = (v10 - 1) & (v24 + v11);
        v12 = (int *)(v8 + 8LL * v11);
        v13 = *v12;
        if ( (unsigned int)*(_QWORD *)(a4 + 8LL * v7) == *v12 )
          goto LABEL_5;
        ++v24;
      }
      break;
    }
LABEL_5:
    if ( v12 == (int *)(v8 + 8 * v10) )
      break;
    v14 = *(_QWORD *)(a4 + 8LL * (v7 + 1));
    v15 = (a2[45].m128i_i64[0] - a2[44].m128i_i64[1]) >> 4;
    if ( v15 > (unsigned int)v14 )
    {
      v16 = (_BYTE *)sub_A08720((__int64)a2, v14);
      goto LABEL_24;
    }
    if ( (unsigned int)v14 < a2->m128i_i32[2] )
    {
      v16 = *(_BYTE **)(a2->m128i_i64[0] + 8LL * (unsigned int)v14);
      if ( v16 )
        goto LABEL_9;
    }
    v29 = (unsigned int)*(_QWORD *)(a4 + 8LL * (v7 + 1));
    if ( ((a2[46].m128i_i64[1] - a2[46].m128i_i64[0]) >> 3) + v15 <= (unsigned int)v14 )
    {
      v16 = (_BYTE *)sub_A07560((__int64)a2, v14);
LABEL_24:
      if ( !v16 )
        goto LABEL_20;
      goto LABEL_9;
    }
    v27 = *(_QWORD *)(a4 + 8LL * (v7 + 1));
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    sub_A04120((__int64 *)&v34, 0);
    sub_A0FFA0(a2, v27, (__int64)&v34, v27);
    v16 = 0;
    sub_A10370(a2, (__int64)&v34, v17, v18, v19);
    if ( (unsigned int)v27 < a2->m128i_i32[2] )
      v16 = *(_BYTE **)(a2->m128i_i64[0] + 8 * v29);
    v32[0] = v40;
    v32[1] = v41;
    v32[2] = v42;
    v32[3] = v43;
    v33[0] = v36;
    v33[1] = v37;
    v33[2] = v38;
    v33[3] = v39;
    sub_A01C60(v33, v32);
    v20 = (__int64)v34;
    if ( !v34 )
      goto LABEL_24;
    v28 = v43 + 8;
    v21 = (_QWORD *)v39;
    if ( v43 + 8 > v39 )
    {
      do
      {
        v30 = v21;
        j_j___libc_free_0(*v21, 512);
        v21 = v30 + 1;
      }
      while ( v28 > (unsigned __int64)(v30 + 1) );
      v20 = (__int64)v34;
    }
    j_j___libc_free_0(v20, 8 * v35);
    if ( !v16 )
    {
LABEL_20:
      BYTE1(v38) = 1;
      v22 = "Invalid metadata attachment: expect fwd ref to MDNode";
      goto LABEL_21;
    }
LABEL_9:
    if ( (unsigned __int8)(*v16 - 5) > 0x1Fu )
      goto LABEL_20;
    v7 += 2;
    sub_B994D0(a3, (unsigned int)v12[1], v16);
    if ( a5 == v7 )
      goto LABEL_11;
  }
  BYTE1(v38) = 1;
  v22 = "Invalid ID";
LABEL_21:
  v34 = v22;
  LOBYTE(v38) = 3;
  sub_A01DB0(a1, (__int64)&v34);
  return a1;
}
