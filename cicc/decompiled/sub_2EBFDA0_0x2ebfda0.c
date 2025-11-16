// Function: sub_2EBFDA0
// Address: 0x2ebfda0
//
unsigned __int64 __fastcall sub_2EBFDA0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned int v5; // ebx
  __int64 v6; // r13
  unsigned __int16 *v7; // rdx
  __int64 v8; // rax
  bool v9; // sf
  __int64 v10; // rax
  unsigned __int16 *v11; // rcx
  char *v12; // rax
  __int64 v13; // rdx
  int v14; // ecx
  __int32 v15; // eax
  __int64 v16; // rax
  _QWORD *v17; // r15
  __int64 *v18; // r9
  _QWORD *v19; // rax
  __int64 *v20; // r9
  __int64 v21; // r8
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rdx
  __int32 v25; // eax
  _QWORD *v27; // [rsp+8h] [rbp-B8h]
  __int64 *v28; // [rsp+8h] [rbp-B8h]
  __int64 *v29; // [rsp+10h] [rbp-B0h]
  __int64 v30; // [rsp+10h] [rbp-B0h]
  __int64 v31; // [rsp+10h] [rbp-B0h]
  __int32 v32; // [rsp+18h] [rbp-A8h]
  __int64 v33; // [rsp+18h] [rbp-A8h]
  int v34; // [rsp+24h] [rbp-9Ch]
  __int64 v35; // [rsp+38h] [rbp-88h] BYREF
  __int64 v36; // [rsp+40h] [rbp-80h] BYREF
  __int64 v37; // [rsp+48h] [rbp-78h]
  __int64 v38; // [rsp+50h] [rbp-70h]
  __m128i v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+70h] [rbp-50h]
  __int64 v41; // [rsp+78h] [rbp-48h]
  __int64 v42; // [rsp+80h] [rbp-40h]

  result = a1[61];
  v34 = (__int64)(a1[62] - result) >> 3;
  if ( v34 )
  {
    v5 = 0;
    v6 = 0;
    v7 = (unsigned __int16 *)a1[61];
    v8 = *(unsigned int *)(result + 4);
    v9 = (int)v8 < 0;
    if ( (_DWORD)v8 )
      goto LABEL_3;
    while ( 1 )
    {
      v15 = *v7;
LABEL_16:
      v39.m128i_i32[0] = v15;
      ++v5;
      v39.m128i_i64[1] = -1;
      v40 = -1;
      result = sub_2EBFD60(a2 + 23, &v39);
      if ( v34 == v5 )
        break;
      while ( 1 )
      {
        v6 = v5;
        v7 = (unsigned __int16 *)(a1[61] + 8LL * v5);
        v8 = *((unsigned int *)v7 + 1);
        v9 = (int)v8 < 0;
        if ( !(_DWORD)v8 )
          break;
LABEL_3:
        if ( v9 )
          v10 = *(_QWORD *)(a1[7] + 16 * (v8 & 0x7FFFFFFF) + 8);
        else
          v10 = *(_QWORD *)(a1[38] + 8 * v8);
        while ( v10 )
        {
          if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 && (*(_BYTE *)(v10 + 4) & 8) == 0 )
          {
            v36 = 0;
            v35 = 0;
            v16 = *(_QWORD *)(a4 + 8);
            v37 = 0;
            v38 = 0;
            v32 = *((_DWORD *)v7 + 1);
            v17 = (_QWORD *)a2[4];
            v18 = (__int64 *)a2[7];
            v39.m128i_i64[0] = 0;
            v29 = v18;
            v19 = sub_2E7B380(v17, v16 - 800, (unsigned __int8 **)&v39, 0);
            v20 = v29;
            v21 = (__int64)v19;
            if ( v39.m128i_i64[0] )
            {
              v27 = v19;
              sub_B91220((__int64)&v39, v39.m128i_i64[0]);
              v21 = (__int64)v27;
              v20 = v29;
            }
            v28 = v20;
            v30 = v21;
            sub_2E31040(a2 + 5, v21);
            v22 = v30;
            v23 = *(_QWORD *)v30;
            v24 = *v28;
            *(_QWORD *)(v30 + 8) = v28;
            v24 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v30 = v24 | v23 & 7;
            *(_QWORD *)(v24 + 8) = v30;
            *v28 = v30 | *v28 & 7;
            if ( v37 )
            {
              sub_2E882B0(v30, (__int64)v17, v37);
              v22 = v30;
            }
            if ( v38 )
            {
              v31 = v22;
              sub_2E88680(v22, (__int64)v17, v38);
              v22 = v31;
            }
            v39.m128i_i64[0] = 0x10000000;
            v39.m128i_i32[2] = v32;
            v33 = v22;
            v40 = 0;
            v41 = 0;
            v42 = 0;
            sub_2E8EAD0(v22, (__int64)v17, &v39);
            v25 = *(_DWORD *)(a1[61] + 8 * v6);
            v39.m128i_i64[0] = 0;
            v40 = 0;
            v39.m128i_i32[2] = v25;
            v41 = 0;
            v42 = 0;
            sub_2E8EAD0(v33, (__int64)v17, &v39);
            if ( v36 )
              sub_B91220((__int64)&v36, v36);
            if ( v35 )
              sub_B91220((__int64)&v35, v35);
            v15 = *(unsigned __int16 *)(a1[61] + 8 * v6);
            goto LABEL_16;
          }
          v10 = *(_QWORD *)(v10 + 32);
        }
        v11 = (unsigned __int16 *)a1[62];
        v12 = (char *)(v7 + 4);
        if ( v11 != v7 + 4 )
        {
          v13 = ((char *)v11 - v12) >> 3;
          if ( (char *)v11 - v12 <= 0 )
          {
            v12 = (char *)a1[62];
          }
          else
          {
            do
            {
              v14 = *(_DWORD *)v12;
              v12 += 8;
              *((_DWORD *)v12 - 4) = v14;
              *((_DWORD *)v12 - 3) = *((_DWORD *)v12 - 1);
              --v13;
            }
            while ( v13 );
            v12 = (char *)a1[62];
          }
        }
        result = (unsigned __int64)(v12 - 8);
        --v34;
        a1[62] = result;
        if ( v34 == v5 )
          return result;
      }
    }
  }
  return result;
}
