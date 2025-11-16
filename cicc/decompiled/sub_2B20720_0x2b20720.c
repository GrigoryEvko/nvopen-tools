// Function: sub_2B20720
// Address: 0x2b20720
//
__int64 __fastcall sub_2B20720(__int64 a1, __int64 a2)
{
  unsigned int v3; // edx
  unsigned int v4; // r13d
  unsigned int v5; // r15d
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 *v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // rdx
  unsigned int v11; // r15d
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 *v14; // rcx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 result; // rax
  unsigned __int64 v18; // rax
  __int64 *v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 *v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __int64 v26; // rbx
  unsigned __int8 v27; // [rsp+Fh] [rbp-181h]
  unsigned __int64 v28; // [rsp+10h] [rbp-180h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-178h]
  __m128i v30; // [rsp+20h] [rbp-170h] BYREF
  __int64 v31; // [rsp+30h] [rbp-160h]
  __int64 v32; // [rsp+38h] [rbp-158h]
  __int64 v33; // [rsp+40h] [rbp-150h]
  __int64 v34; // [rsp+48h] [rbp-148h]
  __int64 v35; // [rsp+50h] [rbp-140h]
  __int64 v36; // [rsp+58h] [rbp-138h]
  __int16 v37; // [rsp+60h] [rbp-130h]
  __m128i v38; // [rsp+70h] [rbp-120h] BYREF
  __int64 v39; // [rsp+80h] [rbp-110h]
  __int64 v40; // [rsp+88h] [rbp-108h]
  __int64 v41; // [rsp+90h] [rbp-100h]
  __int64 v42; // [rsp+98h] [rbp-F8h]
  __int64 v43; // [rsp+A0h] [rbp-F0h]
  __int64 v44; // [rsp+A8h] [rbp-E8h]
  __int16 v45; // [rsp+B0h] [rbp-E0h]
  __m128i v46; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v47; // [rsp+D0h] [rbp-C0h]
  __int64 v48; // [rsp+D8h] [rbp-B8h]
  __int64 v49; // [rsp+E0h] [rbp-B0h]
  __int64 v50; // [rsp+E8h] [rbp-A8h]
  __int64 v51; // [rsp+F0h] [rbp-A0h]
  __int64 v52; // [rsp+F8h] [rbp-98h]
  __int16 v53; // [rsp+100h] [rbp-90h]
  __m128i v54; // [rsp+110h] [rbp-80h] BYREF
  __int64 v55; // [rsp+120h] [rbp-70h]
  __int64 v56; // [rsp+128h] [rbp-68h]
  __int64 v57; // [rsp+130h] [rbp-60h]
  __int64 v58; // [rsp+138h] [rbp-58h]
  __int64 v59; // [rsp+140h] [rbp-50h]
  __int64 v60; // [rsp+148h] [rbp-48h]
  __int16 v61; // [rsp+150h] [rbp-40h]

  v3 = **(_DWORD **)(a1 + 8);
  v4 = **(_DWORD **)(a1 + 16);
  if ( (unsigned int)(**(_DWORD **)a1 - 365) <= 1 )
  {
    v38.m128i_i32[2] = **(_DWORD **)(a1 + 8);
    if ( v3 > 0x40 )
    {
      sub_C43690((__int64)&v38, 0, 0);
      v3 = v38.m128i_u32[2];
    }
    else
    {
      v38.m128i_i64[0] = 0;
    }
    if ( v4 != v3 )
    {
      if ( v4 > 0x3F || v3 > 0x40 )
        sub_C43C90(&v38, v4, v3);
      else
        v38.m128i_i64[0] |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v4 + 64 - (unsigned __int8)v3) << v4;
    }
    v21 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL);
    v53 = 257;
    v46 = (__m128i)v21;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v51 = 0;
    v52 = 0;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v22 = *(__int64 **)(a2 - 8);
    else
      v22 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    result = sub_9AC230(*v22, (__int64)&v38, &v46, 0);
    if ( (_BYTE)result )
    {
      v25 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL);
      v55 = 0;
      v54 = (__m128i)v25;
      v56 = 0;
      v57 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      v61 = 257;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v26 = *(_QWORD *)(a2 - 8);
      else
        v26 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      result = sub_9AC230(*(_QWORD *)(v26 + 32), (__int64)&v38, &v54, 0);
    }
    if ( v38.m128i_i32[2] > 0x40u )
    {
      v20 = v38.m128i_i64[0];
      if ( v38.m128i_i64[0] )
        goto LABEL_27;
    }
  }
  else
  {
    v29 = **(_DWORD **)(a1 + 8);
    v5 = v4 - 1;
    v6 = v3 - v4;
    if ( v3 > 0x40 )
    {
      sub_C43690((__int64)&v28, 0, 0);
      v3 = v29;
    }
    else
    {
      v28 = 0;
    }
    if ( v5 != v3 )
    {
      if ( v5 > 0x3F || v3 > 0x40 )
        sub_C43C90(&v28, v5, v3);
      else
        v28 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v4 + 63 - (unsigned __int8)v3) << v5;
    }
    v7 = *(_QWORD *)(a1 + 24);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v8 = *(__int64 **)(a2 - 8);
    else
      v8 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v9 = sub_9AF8B0(*v8, *(_QWORD *)(v7 + 3344), 0, *(_QWORD *)(v7 + 3328), 0, *(_QWORD *)(v7 + 3320), 1);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v10 = *(_QWORD *)(a2 - 8);
    else
      v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v11 = sub_9AF8B0(
            *(_QWORD *)(v10 + 32),
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL),
            0,
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3328LL),
            0,
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3320LL),
            1);
    if ( v6 > v9 )
      goto LABEL_24;
    if ( v6 == v9 )
      goto LABEL_60;
    v12 = *(_QWORD *)(a1 + 24);
    v31 = 0;
    v13 = *(_QWORD *)(v12 + 3344);
    v32 = 0;
    v33 = 0;
    v30 = (__m128i)v13;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 257;
    v14 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
        ? *(__int64 **)(a2 - 8)
        : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( (unsigned __int8)sub_9AC470(*v14, &v30, 0) )
    {
LABEL_60:
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL);
      v45 = 257;
      v38 = (__m128i)v18;
      v39 = 0;
      v40 = 0;
      v41 = 0;
      v42 = 0;
      v43 = 0;
      v44 = 0;
      v19 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
          ? *(__int64 **)(a2 - 8)
          : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( !(unsigned __int8)sub_9AC230(*v19, (__int64)&v28, &v38, 0) )
        goto LABEL_24;
    }
    if ( v6 > v11 )
    {
LABEL_24:
      result = 0;
    }
    else
    {
      if ( v6 == v11 )
        goto LABEL_45;
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL);
      v47 = 0;
      v46 = (__m128i)v15;
      v48 = 0;
      v49 = 0;
      v50 = 0;
      v51 = 0;
      v52 = 0;
      v53 = 257;
      v16 = sub_986520(a2);
      if ( (unsigned __int8)sub_9AC470(*(_QWORD *)(v16 + 32), &v46, 0) )
      {
LABEL_45:
        v23 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3344LL);
        v55 = 0;
        v54 = (__m128i)v23;
        v56 = 0;
        v57 = 0;
        v58 = 0;
        v59 = 0;
        v60 = 0;
        v61 = 257;
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v24 = *(_QWORD *)(a2 - 8);
        else
          v24 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        result = sub_9AC230(*(_QWORD *)(v24 + 32), (__int64)&v28, &v54, 0);
      }
      else
      {
        result = 1;
      }
    }
    if ( v29 > 0x40 )
    {
      v20 = v28;
      if ( v28 )
      {
LABEL_27:
        v27 = result;
        j_j___libc_free_0_0(v20);
        return v27;
      }
    }
  }
  return result;
}
