// Function: sub_2AB3D90
// Address: 0x2ab3d90
//
char __fastcall sub_2AB3D90(__int64 a1, __int64 a2, __int64 a3, __int32 a4)
{
  __int64 v7; // rax
  __int64 v8; // r10
  int v9; // edi
  bool v10; // zf
  __int64 v11; // r14
  __int64 v12; // r15
  unsigned int v13; // esi
  unsigned int v14; // r13d
  int v15; // r11d
  int v16; // edx
  __int64 v17; // rcx
  char result; // al
  char v19; // al
  char v20; // r10
  __m128i v21; // xmm1
  __int32 v22; // r8d
  __int64 v23; // r9
  char v24; // r10
  bool v25; // sf
  bool v26; // of
  char v27; // al
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  int v30; // edx
  int v31; // ebx
  signed __int64 v32; // rax
  int v33; // edx
  bool v34; // sf
  bool v35; // of
  signed __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  char v39; // [rsp+18h] [rbp-78h]
  __m128i v40; // [rsp+20h] [rbp-70h] BYREF
  __int64 v41; // [rsp+30h] [rbp-60h] BYREF
  int v42; // [rsp+38h] [rbp-58h]
  __int64 v43; // [rsp+40h] [rbp-50h] BYREF
  int v44; // [rsp+48h] [rbp-48h]
  __m128i v45; // [rsp+50h] [rbp-40h] BYREF

  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_DWORD *)(a2 + 16);
  v10 = *(_BYTE *)(v7 + 8) == 0;
  v11 = *(_QWORD *)(a3 + 8);
  v40 = _mm_loadu_si128((const __m128i *)(a2 + 8));
  v12 = *(_QWORD *)(a3 + 16);
  v13 = *(_DWORD *)a2;
  v14 = *(_DWORD *)a3;
  if ( !v10 )
  {
    v15 = *(_DWORD *)(v7 + 4);
    if ( *(_BYTE *)(a2 + 4) )
      v13 *= v15;
    if ( *(_BYTE *)(a3 + 4) )
      v14 *= v15;
  }
  if ( *(_DWORD *)(v7 + 992) != 2 )
  {
    v38 = a1;
    v19 = sub_DFE310(*(_QWORD *)(a1 + 32));
    v20 = 0;
    if ( !v19 )
    {
      v20 = *(_BYTE *)(a2 + 4);
      if ( v20 )
        v20 = *(_BYTE *)(a3 + 4) ^ 1;
    }
    if ( a4 )
    {
      v28 = *(_QWORD *)(a2 + 32);
      v45.m128i_i32[0] = a4;
      v45.m128i_i64[1] = v38;
      v39 = v20;
      v29 = sub_2AA91C0(v45.m128i_i32, v13, v40.m128i_i64[0], v40.m128i_i64[1], *(_QWORD *)(a2 + 24), v28);
      v31 = v30;
      v37 = v29;
      v32 = sub_2AA91C0(v45.m128i_i32, v14, v11, v12, *(_QWORD *)(a3 + 24), *(_QWORD *)(a3 + 32));
      if ( !v39 )
      {
        v35 = __OFSUB__(v31, v33);
        v34 = v31 - v33 < 0;
        if ( v31 == v33 )
        {
          v35 = __OFSUB__(v37, v32);
          v34 = v37 - v32 < 0;
        }
        return v34 ^ v35;
      }
      if ( v31 == v33 )
        v27 = v37 > v32;
      else
        v27 = v33 < v31;
    }
    else
    {
      v45.m128i_i64[0] = v11;
      v43 = v13;
      v44 = 0;
      v45.m128i_i64[1] = v12;
      sub_2AA9150((__int64)&v45, (__int64)&v43);
      v21 = _mm_load_si128(&v40);
      v41 = v14;
      v42 = 0;
      v45 = v21;
      sub_2AA9150((__int64)&v45, (__int64)&v41);
      if ( !v24 )
      {
        if ( v22 == v45.m128i_i32[2] )
          return v23 > v45.m128i_i64[0];
        else
          return v45.m128i_i32[2] < v22;
      }
      v26 = __OFSUB__(v22, v45.m128i_i32[2]);
      v25 = v22 - v45.m128i_i32[2] < 0;
      if ( v22 == v45.m128i_i32[2] )
      {
        v26 = __OFSUB__(v23, v45.m128i_i64[0]);
        v25 = v23 - v45.m128i_i64[0] < 0;
      }
      v27 = v25 ^ v26;
    }
    return v27 ^ 1;
  }
  v16 = *(_DWORD *)(a3 + 16);
  v17 = *(_QWORD *)(a3 + 8);
  result = 1;
  if ( v16 != v9 )
  {
    if ( v16 > v9 )
      return result;
    return v13 > v14 && v16 == v9 && v17 == v8;
  }
  if ( v17 <= v8 )
    return v13 > v14 && v16 == v9 && v17 == v8;
  return result;
}
