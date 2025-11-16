// Function: sub_31C2000
// Address: 0x31c2000
//
__int64 __fastcall sub_31C2000(__int64 a1, __int64 *a2, unsigned __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 *v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 *v9; // r13
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  _BYTE *v25; // rsi
  _BYTE *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 i; // rbx
  __int64 v32; // r15
  __m128i v33; // xmm0
  __int64 v34; // rsi
  __int8 v35; // dl
  __int64 v36; // rax
  bool v37; // zf
  __int64 v38; // rax
  int v39; // edx
  int v40; // r9d
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rbx
  __int64 v44; // rdx
  __int64 j; // r13
  __int64 v46; // r15
  __m128i v47; // xmm3
  __int64 v48; // rsi
  __int8 v49; // cl
  __int8 v50; // dl
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-B0h]
  __int64 *v54; // [rsp+10h] [rbp-A0h]
  __m128i v56; // [rsp+20h] [rbp-90h] BYREF
  __m128i v57; // [rsp+30h] [rbp-80h] BYREF
  __m128i v58; // [rsp+40h] [rbp-70h] BYREF
  __m128i v59; // [rsp+50h] [rbp-60h] BYREF
  __m128i v60; // [rsp+60h] [rbp-50h] BYREF
  __m128i v61; // [rsp+70h] [rbp-40h]

  v54 = &a2[a3];
  v4 = *(_QWORD *)(a1 + 72);
  v5 = (__int64)(8 * a3) >> 5;
  v52 = (__int64)(8 * a3) >> 3;
  if ( !v4 )
    goto LABEL_10;
  v6 = a2;
  v7 = sub_318B4F0(v4);
  if ( v5 <= 0 )
  {
    v38 = v52;
    v6 = a2;
LABEL_68:
    if ( v38 != 2 )
    {
      if ( v38 != 3 )
      {
        if ( v38 != 1 )
          goto LABEL_10;
        goto LABEL_71;
      }
      if ( v7 != sub_318B4F0(*v6) )
        goto LABEL_9;
      ++v6;
    }
    if ( v7 != sub_318B4F0(*v6) )
      goto LABEL_9;
    ++v6;
LABEL_71:
    if ( v7 == sub_318B4F0(*v6) )
      goto LABEL_10;
    goto LABEL_9;
  }
  v8 = v5;
  while ( v7 == sub_318B4F0(*v6) )
  {
    if ( v7 != sub_318B4F0(v6[1]) )
    {
      if ( v6 + 1 == v54 )
        goto LABEL_10;
      return 0;
    }
    if ( v7 != sub_318B4F0(v6[2]) )
    {
      if ( v6 + 2 == v54 )
        goto LABEL_10;
      return 0;
    }
    if ( v7 != sub_318B4F0(v6[3]) )
    {
      if ( v6 + 3 == v54 )
        goto LABEL_10;
      return 0;
    }
    v6 += 4;
    if ( !--v8 )
    {
      v38 = v54 - v6;
      goto LABEL_68;
    }
  }
LABEL_9:
  if ( v6 != v54 )
    return 0;
LABEL_10:
  if ( !*(_QWORD *)(a1 + 248) )
    *(_QWORD *)(a1 + 248) = sub_318B4F0(*a2);
  v9 = a2;
  if ( v5 <= 0 )
  {
LABEL_22:
    if ( v52 != 2 )
    {
      if ( v52 != 3 )
      {
        if ( v52 != 1 )
          goto LABEL_26;
LABEL_25:
        if ( *(_QWORD *)(a1 + 248) == sub_318B4F0(*v9) )
          goto LABEL_26;
        goto LABEL_19;
      }
      if ( *(_QWORD *)(a1 + 248) != sub_318B4F0(*v9) )
      {
LABEL_19:
        if ( v9 == v54 )
          goto LABEL_26;
        return 0;
      }
      ++v9;
    }
    if ( *(_QWORD *)(a1 + 248) == sub_318B4F0(*v9) )
    {
      ++v9;
      goto LABEL_25;
    }
    goto LABEL_19;
  }
  while ( 1 )
  {
    if ( *(_QWORD *)(a1 + 248) != sub_318B4F0(*v9) )
      goto LABEL_19;
    if ( *(_QWORD *)(a1 + 248) != sub_318B4F0(v9[1]) )
    {
      if ( v9 + 1 == v54 )
        goto LABEL_26;
      return 0;
    }
    if ( *(_QWORD *)(a1 + 248) != sub_318B4F0(v9[2]) )
    {
      if ( v9 + 2 == v54 )
        goto LABEL_26;
      return 0;
    }
    if ( *(_QWORD *)(a1 + 248) != sub_318B4F0(v9[3]) )
      break;
    v9 += 4;
    if ( !--v5 )
    {
      v52 = v54 - v9;
      goto LABEL_22;
    }
  }
  if ( v9 + 3 != v54 )
    return 0;
LABEL_26:
  v11 = sub_31C01E0(a1, a2, a3);
  if ( v11 == 2 )
  {
    sub_31BBC70(a1 + 40, a2, a3);
    sub_31C08E0(a1, a2, a3);
    v27 = sub_31BFEB0((__int64)a2, a3, 1);
    v28 = *a2;
    v30 = v29;
    for ( i = v27; v30 != i; i += 8 )
    {
      v32 = *(_QWORD *)i;
      if ( sub_B445A0(*(_QWORD *)(v28 + 16), *(_QWORD *)(*(_QWORD *)i + 16LL)) )
        v28 = v32;
    }
    sub_318B480((__int64)&v56, v28);
    v33 = _mm_loadu_si128(&v56);
    v61 = _mm_loadu_si128(&v57);
    v60 = v33;
    sub_371B2F0(&v60);
    v34 = v60.m128i_i64[1];
    v13 = v61.m128i_u8[0];
    v35 = v61.m128i_i8[1];
    v36 = v61.m128i_i64[1];
    *(_QWORD *)(a1 + 176) = v60.m128i_i64[0];
    v37 = *(_BYTE *)(a1 + 208) == 0;
    *(_QWORD *)(a1 + 184) = v34;
    *(_BYTE *)(a1 + 192) = v13;
    *(_BYTE *)(a1 + 193) = v35;
    *(_QWORD *)(a1 + 200) = v36;
    if ( v37 )
      *(_BYTE *)(a1 + 208) = 1;
    return sub_31C1C20((__int64 *)a1, a2, a3, v13, v15, v16);
  }
  else
  {
    if ( v11 <= 2 )
    {
      if ( !v11 )
      {
        if ( !*(_BYTE *)(a1 + 208) )
        {
          v41 = sub_31BFEB0((__int64)a2, a3, 1);
          v42 = *a2;
          v43 = v41;
          for ( j = v44; j != v43; v43 += 8 )
          {
            v46 = *(_QWORD *)v43;
            if ( sub_B445A0(*(_QWORD *)(v42 + 16), *(_QWORD *)(*(_QWORD *)v43 + 16LL)) )
              v42 = v46;
          }
          sub_318B480((__int64)&v58, v42);
          v47 = _mm_loadu_si128(&v59);
          v60 = _mm_loadu_si128(&v58);
          v61 = v47;
          sub_371B2F0(&v60);
          v48 = v60.m128i_i64[1];
          v49 = v61.m128i_i8[0];
          v50 = v61.m128i_i8[1];
          v51 = v61.m128i_i64[1];
          *(_QWORD *)(a1 + 176) = v60.m128i_i64[0];
          v37 = *(_BYTE *)(a1 + 208) == 0;
          *(_QWORD *)(a1 + 184) = v48;
          *(_BYTE *)(a1 + 192) = v49;
          *(_BYTE *)(a1 + 193) = v50;
          *(_QWORD *)(a1 + 200) = v51;
          if ( v37 )
            *(_BYTE *)(a1 + 208) = 1;
        }
        v14 = sub_31BBC70(a1 + 40, a2, a3);
        v17 = v14;
        v18 = v12;
        if ( v12 )
          v18 = sub_318B4B0(v12);
        if ( v14 == v18 )
          return sub_31C1C20((__int64 *)a1, a2, a3, v13, v15, v16);
        while ( 1 )
        {
          v19 = *(unsigned int *)(a1 + 64);
          v20 = *(_QWORD *)(a1 + 48);
          if ( !(_DWORD)v19 )
            goto LABEL_93;
          v21 = (v19 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v17 != *v22 )
            break;
LABEL_37:
          if ( v22 == (__int64 *)(v20 + 16 * v19) )
            goto LABEL_93;
          v24 = v22[1];
          if ( !*(_DWORD *)(v24 + 20) )
          {
            v60.m128i_i64[0] = v22[1];
            v25 = *(_BYTE **)(a1 + 16);
            if ( v25 == *(_BYTE **)(a1 + 24) )
            {
              sub_31C0410(a1 + 8, v25, &v60);
              v26 = *(_BYTE **)(a1 + 16);
            }
            else
            {
              if ( v25 )
              {
                *(_QWORD *)v25 = v24;
                v25 = *(_BYTE **)(a1 + 16);
              }
              v26 = v25 + 8;
              *(_QWORD *)(a1 + 16) = v26;
            }
            sub_31BFEC0(*(_QWORD *)(a1 + 8), ((__int64)&v26[-*(_QWORD *)(a1 + 8)] >> 3) - 1, 0, *((_QWORD *)v26 - 1));
          }
          v17 = sub_318B4B0(v17);
          if ( v17 == v18 )
            return sub_31C1C20((__int64 *)a1, a2, a3, v13, v15, v16);
        }
        v39 = 1;
        while ( v23 != -4096 )
        {
          v40 = v39 + 1;
          v21 = (v19 - 1) & (v39 + v21);
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v17 == *v22 )
            goto LABEL_37;
          v39 = v40;
        }
LABEL_93:
        BUG();
      }
      if ( v11 != 1 )
        goto LABEL_93;
      return 0;
    }
    if ( v11 != 3 )
      goto LABEL_93;
    return 1;
  }
}
