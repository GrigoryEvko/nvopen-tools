// Function: sub_2BEC0F0
// Address: 0x2bec0f0
//
__int64 __fastcall sub_2BEC0F0(__int64 a1)
{
  unsigned int v2; // r13d
  int v3; // eax
  unsigned __int64 *v5; // rbx
  __int64 v6; // rsi
  int v7; // ecx
  void *v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  unsigned __int64 *v15; // rdi
  int v16; // edx
  int v17; // ecx
  void *v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // r14
  int v23; // eax
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rcx
  __m128i v27; // xmm4
  __m128i v28; // xmm2
  __m128i v29; // xmm3
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // rbx
  __int64 v35; // rax
  __m128i v36; // xmm0
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __m128i *v39; // rsi
  __m128i v40; // xmm0
  __m128i v41; // xmm1
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __m128i *v45; // rsi
  __int64 v46; // rsi
  unsigned __int64 v47; // rax
  __int64 v48; // rdx
  _QWORD *v49; // [rsp+8h] [rbp-D8h]
  __m128i v50; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v51; // [rsp+20h] [rbp-C0h]
  __m128i v52; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v53; // [rsp+40h] [rbp-A0h]
  __m128i v54; // [rsp+50h] [rbp-90h] BYREF
  __m128i v55; // [rsp+60h] [rbp-80h] BYREF
  __m128i v56; // [rsp+70h] [rbp-70h] BYREF
  __m128i v57; // [rsp+80h] [rbp-60h] BYREF
  __m128i v58; // [rsp+90h] [rbp-50h] BYREF
  __m128i v59; // [rsp+A0h] [rbp-40h] BYREF

  if ( !*(_DWORD *)(a1 + 152) )
  {
    v2 = sub_2BE0030(a1);
    if ( (_BYTE)v2 )
    {
      v14 = *(_DWORD *)a1;
      v15 = *(unsigned __int64 **)(a1 + 256);
      v16 = *(_DWORD *)a1 & 8;
      v17 = *(_DWORD *)a1 & 1;
      if ( (v14 & 0x10) != 0 )
      {
        if ( v17 )
        {
          v57.m128i_i64[0] = *(_QWORD *)(a1 + 384);
          if ( v16 )
          {
            v58.m128i_i64[1] = (__int64)sub_2BDC740;
            v18 = sub_2BDB940;
          }
          else
          {
            v58.m128i_i64[1] = (__int64)sub_2BDC7C0;
            v18 = sub_2BDB910;
          }
        }
        else if ( v16 )
        {
          v57.m128i_i64[0] = *(_QWORD *)(a1 + 384);
          v58.m128i_i64[1] = (__int64)sub_2BDB6F0;
          v18 = sub_2BDB8E0;
        }
        else
        {
          v58.m128i_i64[1] = (__int64)sub_2BDB6C0;
          v18 = sub_2BDB6E0;
        }
      }
      else if ( v17 )
      {
        v57.m128i_i64[0] = *(_QWORD *)(a1 + 384);
        if ( v16 )
        {
          v58.m128i_i64[1] = (__int64)sub_2BDDA70;
          v18 = sub_2BDB8B0;
        }
        else
        {
          v58.m128i_i64[1] = (__int64)sub_2BDDB00;
          v18 = sub_2BDB880;
        }
      }
      else if ( v16 )
      {
        v57.m128i_i64[0] = *(_QWORD *)(a1 + 384);
        v58.m128i_i64[1] = (__int64)sub_2BDBB70;
        v18 = sub_2BDB850;
      }
      else
      {
        v58.m128i_i64[1] = (__int64)sub_2BDBBD0;
        v18 = sub_2BDB6B0;
      }
      v58.m128i_i64[0] = (__int64)v18;
      v19 = sub_2BE0EB0(v15, &v57);
      v20 = *(_QWORD *)(a1 + 256);
      v54.m128i_i64[1] = v19;
      v54.m128i_i64[0] = v20;
      v55.m128i_i64[0] = v19;
      sub_2BE3490((unsigned __int64 *)(a1 + 304), &v54);
      sub_A17130((__int64)&v57);
      return v2;
    }
  }
  v2 = sub_2BE0770(a1);
  if ( (_BYTE)v2 )
  {
    v5 = *(unsigned __int64 **)(a1 + 256);
    v6 = **(unsigned __int8 **)(a1 + 272);
    v7 = *(_DWORD *)a1 & 8;
    if ( (*(_DWORD *)a1 & 1) != 0 )
    {
      v49 = *(_QWORD **)(a1 + 384);
      if ( v7 )
      {
        v9 = sub_222F790(v49, v6);
        v57.m128i_i8[8] = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 32LL))(
                            v9,
                            (unsigned int)(char)v6);
        v57.m128i_i64[0] = (__int64)v49;
        v58.m128i_i64[1] = (__int64)sub_2BDC5F0;
        v8 = sub_2BDBA00;
      }
      else
      {
        v12 = sub_222F790(v49, v6);
        v57.m128i_i8[8] = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v12 + 32LL))(
                            v12,
                            (unsigned int)(char)v6);
        v57.m128i_i64[0] = (__int64)v49;
        v58.m128i_i64[1] = (__int64)sub_2BDC5C0;
        v8 = sub_2BDB9D0;
      }
    }
    else if ( v7 )
    {
      v13 = *(_QWORD *)(a1 + 384);
      v57.m128i_i8[8] = **(_BYTE **)(a1 + 272);
      v57.m128i_i64[0] = v13;
      v58.m128i_i64[1] = (__int64)sub_2BDB720;
      v8 = sub_2BDB9A0;
    }
    else
    {
      v57.m128i_i8[1] = **(_BYTE **)(a1 + 272);
      v58.m128i_i64[1] = (__int64)sub_2BDB710;
      v8 = sub_2BDB970;
    }
    v58.m128i_i64[0] = (__int64)v8;
    v10 = sub_2BE0EB0(v5, &v57);
    v11 = *(_QWORD *)(a1 + 256);
    v54.m128i_i64[1] = v10;
    v54.m128i_i64[0] = v11;
    v55.m128i_i64[0] = v10;
    sub_2BE3490((unsigned __int64 *)(a1 + 304), &v54);
    sub_A17130((__int64)&v57);
    return v2;
  }
  v3 = *(_DWORD *)(a1 + 152);
  if ( v3 == 4 )
  {
    v2 = sub_2BE0030(a1);
    if ( (_BYTE)v2 )
    {
      v22 = *(_QWORD *)(a1 + 256);
      v23 = sub_2BE08E0(a1, 10);
      v24 = v23;
      if ( (*(_BYTE *)(v22 + 25) & 4) == 0 && (unsigned __int64)v23 < *(_QWORD *)(v22 + 40) )
      {
        v25 = *(_QWORD **)v22;
        v26 = *(_QWORD **)(v22 + 8);
        if ( *(_QWORD **)v22 == v26 )
        {
LABEL_38:
          *(_BYTE *)(v22 + 48) = 1;
          v27 = _mm_loadu_si128(&v56);
          v54.m128i_i32[0] = 3;
          v54.m128i_i64[1] = -1;
          v28 = _mm_loadu_si128(&v54);
          v55.m128i_i64[0] = v24;
          v29 = _mm_loadu_si128(&v55);
          v57 = v28;
          v58 = v29;
          v59 = v27;
          v30 = sub_2BE03F0((unsigned __int64 *)v22, &v57);
          if ( v57.m128i_i32[0] == 11 )
            sub_A17130((__int64)&v58);
          if ( v54.m128i_i32[0] == 11 )
            sub_A17130((__int64)&v55);
          v31 = *(_QWORD *)(a1 + 256);
          v57.m128i_i64[1] = v30;
          v58.m128i_i64[0] = v30;
          v57.m128i_i64[0] = v31;
          sub_2BE3490((unsigned __int64 *)(a1 + 304), &v57);
          return v2;
        }
        while ( v24 != *v25 )
        {
          if ( v26 == ++v25 )
            goto LABEL_38;
        }
      }
      goto LABEL_78;
    }
    v3 = *(_DWORD *)(a1 + 152);
  }
  if ( v3 == 14 )
  {
    v2 = sub_2BE0030(a1);
    if ( (_BYTE)v2 )
    {
      v21 = *(_DWORD *)a1 & 8;
      if ( (*(_DWORD *)a1 & 1) != 0 )
      {
        if ( v21 )
          sub_2BE58D0((_QWORD *)a1);
        else
          sub_2BE4EB0((_QWORD *)a1);
      }
      else if ( v21 )
      {
        sub_2BE4580((_QWORD *)a1);
      }
      else
      {
        sub_2BE3B10((_QWORD *)a1);
      }
      return v2;
    }
    v3 = *(_DWORD *)(a1 + 152);
  }
  if ( v3 != 6 )
  {
LABEL_6:
    if ( v3 != 5 || !(unsigned __int8)sub_2BE0030(a1) )
      return sub_2BEC030((_DWORD *)a1);
    v32 = sub_2BE05D0(*(_QWORD **)(a1 + 256));
    v33 = *(_QWORD *)(a1 + 256);
    v52.m128i_i64[1] = v32;
    v52.m128i_i64[0] = v33;
    v53 = v32;
    sub_2BECC80(a1);
    if ( *(_DWORD *)(a1 + 152) == 8 )
    {
      v2 = sub_2BE0030(a1);
      if ( (_BYTE)v2 )
      {
        sub_2BDEE20(&v57, (_QWORD *)a1);
        *(_QWORD *)(*(_QWORD *)(v52.m128i_i64[0] + 56) + 48 * v53 + 8) = v57.m128i_i64[1];
        v34 = *(_QWORD **)(a1 + 256);
        v54.m128i_i32[0] = 9;
        v54.m128i_i64[1] = -1;
        v53 = v58.m128i_i64[0];
        v35 = v34[1];
        v55.m128i_i64[0] = *(_QWORD *)(v35 - 8);
        v34[1] = v35 - 8;
        v36 = _mm_loadu_si128(&v54);
        v37 = _mm_loadu_si128(&v55);
        v38 = _mm_loadu_si128(&v56);
        v57 = v36;
        v58 = v37;
        v59 = v38;
        v39 = (__m128i *)v34[8];
        if ( v39 == (__m128i *)v34[9] )
        {
          sub_2BE00E0(v34 + 7, v39, &v57);
          v45 = (__m128i *)v34[8];
        }
        else
        {
          if ( v39 )
          {
            *v39 = v36;
            v40 = _mm_loadu_si128(&v58);
            v39[1] = v40;
            v39[2] = _mm_loadu_si128(&v59);
            if ( v57.m128i_i32[0] == 11 )
            {
              v39[2].m128i_i64[0] = 0;
              v41 = _mm_loadu_si128(&v58);
              v58 = v40;
              v39[1] = v41;
              v42 = v59.m128i_i64[0];
              v59.m128i_i64[0] = 0;
              v43 = v39[2].m128i_i64[1];
              v39[2].m128i_i64[0] = v42;
              v44 = v59.m128i_i64[1];
              v59.m128i_i64[1] = v43;
              v39[2].m128i_i64[1] = v44;
            }
            v39 = (__m128i *)v34[8];
          }
          v45 = v39 + 3;
          v34[8] = v45;
        }
        v46 = (__int64)v45->m128i_i64 - v34[7];
        if ( (unsigned __int64)v46 <= 0x493E00 )
        {
          if ( v57.m128i_i32[0] == 11 && v59.m128i_i64[0] )
            ((void (__fastcall *)(__m128i *, __m128i *, __int64))v59.m128i_i64[0])(&v58, &v58, 3);
          if ( v54.m128i_i32[0] == 11 )
          {
            if ( v56.m128i_i64[0] )
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v56.m128i_i64[0])(&v55, &v55, 3);
          }
          *(_QWORD *)(*(_QWORD *)(v52.m128i_i64[0] + 56) + 48 * v53 + 8) = 0xAAAAAAAAAAAAAAABLL * (v46 >> 4) - 1;
          v53 = 0xAAAAAAAAAAAAAAABLL * (v46 >> 4) - 1;
          sub_2BE3450((unsigned __int64 *)(a1 + 304), &v52);
          return v2;
        }
      }
    }
LABEL_78:
    abort();
  }
  if ( !(unsigned __int8)sub_2BE0030(a1) )
  {
    v3 = *(_DWORD *)(a1 + 152);
    goto LABEL_6;
  }
  v47 = sub_2BE04C0(*(unsigned __int64 **)(a1 + 256));
  v48 = *(_QWORD *)(a1 + 256);
  v50.m128i_i64[1] = v47;
  v50.m128i_i64[0] = v48;
  v51 = v47;
  sub_2BECC80(a1);
  if ( *(_DWORD *)(a1 + 152) != 8 )
    goto LABEL_78;
  v2 = sub_2BE0030(a1);
  if ( !(_BYTE)v2 )
    goto LABEL_78;
  sub_2BDEE20(&v57, (_QWORD *)a1);
  *(_QWORD *)(*(_QWORD *)(v50.m128i_i64[0] + 56) + 48 * v51 + 8) = v57.m128i_i64[1];
  v51 = v58.m128i_i64[0];
  sub_2BE3450((unsigned __int64 *)(a1 + 304), &v50);
  return v2;
}
