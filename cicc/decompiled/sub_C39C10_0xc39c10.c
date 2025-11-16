// Function: sub_C39C10
// Address: 0xc39c10
//
__int64 __fastcall sub_C39C10(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v5; // r14d
  __int64 *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // eax
  int v11; // edx
  int v12; // ecx
  unsigned int v13; // eax
  __int64 v14; // rsi
  const __m128i *v15; // rdi
  const __m128i *v16; // rdx
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned int v19; // r14d
  unsigned int v20; // ebx
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 *v24; // rdi
  int v26; // eax
  __int64 v27; // rax
  int v28; // [rsp+14h] [rbp-CCh]
  unsigned int v31; // [rsp+20h] [rbp-C0h]
  unsigned int v32; // [rsp+28h] [rbp-B8h]
  bool v33; // [rsp+2Fh] [rbp-B1h]
  __int64 v34; // [rsp+30h] [rbp-B0h]
  unsigned int v35; // [rsp+38h] [rbp-A8h]
  int v36; // [rsp+38h] [rbp-A8h]
  bool v37; // [rsp+4Fh] [rbp-91h] BYREF
  _QWORD v38[4]; // [rsp+50h] [rbp-90h] BYREF
  __m128i v39; // [rsp+70h] [rbp-70h] BYREF
  __int64 v40; // [rsp+80h] [rbp-60h]
  __int32 v41; // [rsp+88h] [rbp-58h]
  _BYTE src[80]; // [rsp+90h] [rbp-50h] BYREF

  v32 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v28 = 2 * v32;
  v5 = (2 * v32 + 64) >> 6;
  if ( 2 * v32 + 64 > 0x7F )
  {
    if ( v5 <= 4 )
    {
      v33 = 0;
      v6 = (__int64 *)src;
    }
    else
    {
      v6 = (__int64 *)sub_2207820(8LL * v5);
      v33 = v6 != 0;
    }
  }
  else
  {
    v33 = 0;
    v5 = 1;
    v6 = (__int64 *)src;
  }
  v34 = sub_C33900(a1);
  v7 = sub_C337D0(a1);
  v8 = sub_C33930(a2);
  sub_C47530(v6, v34, v8, v7, v7);
  v10 = sub_C45E30(v6, v5, v9);
  v35 = 0;
  v11 = *(_DWORD *)(a1 + 16) + *(_DWORD *)(a2 + 16) + 2;
  v12 = v10;
  *(_DWORD *)(a1 + 16) = v11;
  v13 = v10 + 1;
  if ( !a4 && (*(_BYTE *)(a3 + 20) & 7) != 3 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(const __m128i **)a1;
    v40 = 0;
    v16 = v15;
    LOWORD(v41) = 257;
    if ( v28 != v13 )
    {
      v36 = v12;
      sub_C475D0(v6);
      v16 = *(const __m128i **)a1;
      *(_DWORD *)(a1 + 16) += v36 - v28 + 1;
    }
    v39 = _mm_loadu_si128(v16);
    v40 = v16[1].m128i_i64[0];
    v41 = v16[1].m128i_i32[2];
    v39.m128i_i32[2] = v28 + 1;
    if ( v5 == 1 )
    {
      v27 = *v6;
      *(_QWORD *)a1 = &v39;
      *(_QWORD *)(a1 + 8) = v27;
      sub_C33EB0(v38, (__int64 *)a3);
      sub_C396A0((__int64)v38, &v39, 0, &v37);
      sub_C342A0((__int64)v38, 1u);
      v35 = sub_C376D0(a1, (__int64)v38, 0);
      *v6 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v6;
      *(_QWORD *)a1 = &v39;
      sub_C33EB0(v38, (__int64 *)a3);
      sub_C396A0((__int64)v38, &v39, 0, &v37);
      sub_C342A0((__int64)v38, 1u);
      v35 = sub_C376D0(a1, (__int64)v38, 0);
    }
    *(_QWORD *)(a1 + 8) = v14;
    *(_QWORD *)a1 = v15;
    v31 = sub_C45E30(v6, v5, v17) + 1;
    sub_C338F0((__int64)v38);
    v11 = *(_DWORD *)(a1 + 16);
    v13 = v31;
  }
  *(_DWORD *)(a1 + 16) = ~v32 + v11;
  if ( v13 > v32 )
  {
    v18 = v13 - v32;
    v19 = (v13 + 63) >> 6;
    if ( !v19 )
      v19 = 1;
    v20 = v13 - v32;
    v21 = sub_C45DF0(v6, v19);
    if ( v18 <= v21 )
    {
      sub_C48220(v6, v19, v18);
      v35 = v35 != 0;
    }
    else
    {
      if ( v18 == v21 + 1 )
      {
        sub_C48220(v6, v19, v18);
        v35 = 2 - ((v35 == 0) - 1);
        goto LABEL_18;
      }
      if ( v18 <= v19 << 6 )
      {
        v26 = sub_C45D90(v6, v18 - 1);
        v22 = v18;
        v23 = v19;
        v24 = v6;
        if ( v26 )
        {
          sub_C48220(v6, v19, v20);
          v35 = 3;
          goto LABEL_18;
        }
      }
      else
      {
        v22 = v18;
        v23 = v19;
        v24 = v6;
      }
      sub_C48220(v24, v23, v22);
      v35 = 1;
    }
LABEL_18:
    *(_DWORD *)(a1 + 16) += v20;
  }
  sub_C45D30(v34, v6, v7);
  if ( v33 )
    j_j___libc_free_0_0(v6);
  return v35;
}
