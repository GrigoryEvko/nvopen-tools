// Function: sub_1103330
// Address: 0x1103330
//
__int64 __fastcall sub_1103330(__int64 a1, _BYTE *a2, __int64 a3, int a4)
{
  unsigned int v7; // r12d
  int v8; // r12d
  unsigned int v9; // eax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  const __m128i *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // rcx
  __int64 v18; // r12
  __int64 v20; // rsi
  _BYTE *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r14
  unsigned int v24; // ebx
  int v25; // eax
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // r14
  unsigned int v29; // ebx
  __int64 v30; // rax
  bool v31; // zf
  __int64 v32; // rdx
  _BYTE *v33; // rax
  _BYTE *v34; // rbx
  __int64 v35; // r8
  unsigned int v36; // r14d
  int v37; // eax
  bool v38; // al
  _BYTE *v39; // r14
  unsigned int v40; // ebx
  __int64 v41; // rax
  __int64 v42; // rdx
  _BYTE *v43; // rax
  __int64 v44; // rdx
  _BYTE *v45; // rax
  int v46; // eax
  __int64 v47; // rdx
  _BYTE *v48; // rax
  int v49; // eax
  __int64 v50; // r14
  __int64 v51; // rdx
  _BYTE *v52; // rax
  unsigned __int8 *v53; // r8
  unsigned int v54; // r14d
  int v55; // eax
  int v56; // eax
  __int64 v57; // rsi
  bool v58; // r14
  __int64 v59; // rax
  unsigned int v60; // r14d
  int v61; // eax
  __int64 v62; // [rsp+0h] [rbp-B0h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v64; // [rsp+10h] [rbp-A0h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+18h] [rbp-98h]
  __int64 v67; // [rsp+18h] [rbp-98h]
  __int64 v68; // [rsp+18h] [rbp-98h]
  __int64 v69; // [rsp+18h] [rbp-98h]
  int v70; // [rsp+18h] [rbp-98h]
  __int64 v71; // [rsp+18h] [rbp-98h]
  __int64 v72; // [rsp+18h] [rbp-98h]
  __int64 v73; // [rsp+18h] [rbp-98h]
  int v74; // [rsp+18h] [rbp-98h]
  unsigned __int64 v75; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v76; // [rsp+28h] [rbp-88h]
  __m128i v77; // [rsp+30h] [rbp-80h] BYREF
  __m128i v78; // [rsp+40h] [rbp-70h]
  __m128i v79; // [rsp+50h] [rbp-60h]
  __m128i v80; // [rsp+60h] [rbp-50h]
  __int64 v81; // [rsp+70h] [rbp-40h]

  v7 = **(_DWORD **)a1;
  if ( v7 )
  {
    _BitScanReverse(&v7, v7);
    v8 = v7 ^ 0x1F;
    v9 = **(_DWORD **)(a1 + 8);
    v77.m128i_i32[2] = v9;
    if ( v9 > 0x40 )
    {
      sub_C43690((__int64)&v77, 0, 0);
      if ( 31 == v8 )
        goto LABEL_6;
    }
    else
    {
      v77.m128i_i64[0] = 0;
      if ( 31 == v8 )
      {
        v20 = -1;
        goto LABEL_20;
      }
    }
    v9 = v77.m128i_u32[2];
    v10 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v8 + 33);
    if ( v77.m128i_i32[2] <= 0x40u )
    {
      v20 = ~(v10 | v77.m128i_i64[0]);
      goto LABEL_20;
    }
    *(_QWORD *)v77.m128i_i64[0] |= v10;
  }
  else
  {
    v77.m128i_i32[2] = **(_DWORD **)(a1 + 8);
    if ( v77.m128i_i32[2] > 0x40u )
      sub_C43690((__int64)&v77, 0, 0);
    else
      v77.m128i_i64[0] = 0;
    sub_C43C90(&v77, 0, 0xFFFFFFFF);
  }
LABEL_6:
  v9 = v77.m128i_u32[2];
  if ( v77.m128i_i32[2] <= 0x40u )
  {
    v20 = ~v77.m128i_i64[0];
LABEL_20:
    v11 = v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9);
    if ( !v9 )
      v11 = 0;
    goto LABEL_8;
  }
  sub_C43D10((__int64)&v77);
  v9 = v77.m128i_u32[2];
  v11 = v77.m128i_i64[0];
LABEL_8:
  v76 = v9;
  v12 = *(_QWORD **)(a1 + 24);
  v75 = v11;
  if ( **(_QWORD **)(a1 + 16) == *v12 )
    goto LABEL_29;
  v13 = *(const __m128i **)(a1 + 32);
  v77 = _mm_loadu_si128(v13 + 6);
  v78 = _mm_loadu_si128(v13 + 7);
  v79 = _mm_loadu_si128(v13 + 8);
  v80 = _mm_loadu_si128(v13 + 9);
  v14 = v13[10].m128i_i64[0];
  v79.m128i_i64[1] = 0;
  v81 = v14;
  if ( (unsigned __int8)sub_9AC230((__int64)a2, (__int64)&v75, &v77, 0) )
  {
LABEL_29:
    v15 = *(_QWORD *)(a3 + 16);
    if ( v15 )
    {
      if ( !*(_QWORD *)(v15 + 8) && *(_BYTE *)a3 == 44 )
      {
        v21 = *(_BYTE **)(a3 - 64);
        if ( !v21 )
          goto LABEL_77;
        if ( *v21 == 17
          || (v44 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v21 + 1) + 8LL) - 17, (unsigned int)v44 <= 1)
          && *v21 <= 0x15u
          && (v45 = sub_AD7630(*(_QWORD *)(a3 - 64), 0, v44), (v21 = v45) != 0)
          && *v45 == 17 )
        {
          if ( *((_DWORD *)v21 + 8) > 0x40u )
          {
            v70 = *((_DWORD *)v21 + 8);
            if ( v70 - (unsigned int)sub_C444A0((__int64)(v21 + 24)) > 0x40 )
              goto LABEL_11;
            v22 = **((_QWORD **)v21 + 3);
          }
          else
          {
            v22 = *((_QWORD *)v21 + 3);
          }
          if ( a4 == v22 )
          {
            v18 = (__int64)a2;
            if ( a2 == *(_BYTE **)(a3 - 32) )
              goto LABEL_15;
          }
        }
      }
    }
  }
LABEL_11:
  if ( **(_QWORD **)(a1 + 16) != **(_QWORD **)(a1 + 24) )
    goto LABEL_14;
  v16 = *a2;
  v17 = (unsigned int)(a4 - 1);
  if ( *a2 == 57 )
  {
    v18 = *((_QWORD *)a2 - 8);
    if ( !v18 )
      goto LABEL_14;
    v23 = *((_QWORD *)a2 - 4);
    if ( !v23 )
LABEL_77:
      BUG();
    if ( *(_BYTE *)v23 != 17 )
    {
      v66 = v17;
      v32 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v23 + 8) + 8LL) - 17;
      if ( (unsigned int)v32 > 1 || *(_BYTE *)v23 > 0x15u )
        goto LABEL_14;
      v33 = sub_AD7630(v23, 0, v32);
      v17 = v66;
      v23 = (__int64)v33;
      if ( !v33 || *v33 != 17 )
        goto LABEL_46;
    }
    v24 = *(_DWORD *)(v23 + 32);
    if ( v24 <= 0x40 )
    {
      v26 = *(_QWORD *)(v23 + 24);
    }
    else
    {
      v65 = v17;
      v25 = sub_C444A0(v23 + 24);
      v17 = v65;
      if ( v24 - v25 > 0x40 )
        goto LABEL_46;
      v26 = **(_QWORD **)(v23 + 24);
    }
    if ( v17 != v26 )
      goto LABEL_46;
    if ( *(_BYTE *)a3 != 57 )
      goto LABEL_46;
    v34 = *(_BYTE **)(a3 - 64);
    if ( *v34 != 44 )
      goto LABEL_46;
    v35 = *((_QWORD *)v34 - 8);
    if ( *(_BYTE *)v35 == 17 )
    {
      v36 = *(_DWORD *)(v35 + 32);
      if ( v36 <= 0x40 )
      {
        v38 = *(_QWORD *)(v35 + 24) == 0;
      }
      else
      {
        v67 = v17;
        v37 = sub_C444A0(v35 + 24);
        v17 = v67;
        v38 = v36 == v37;
      }
    }
    else
    {
      v50 = *(_QWORD *)(v35 + 8);
      v51 = (unsigned int)*(unsigned __int8 *)(v50 + 8) - 17;
      if ( (unsigned int)v51 > 1 || *(_BYTE *)v35 > 0x15u )
        goto LABEL_46;
      v63 = v17;
      v73 = *((_QWORD *)v34 - 8);
      v52 = sub_AD7630(v73, 0, v51);
      v53 = (unsigned __int8 *)v73;
      v17 = v63;
      if ( !v52 || *v52 != 17 )
      {
        if ( *(_BYTE *)(v50 + 8) != 17 )
          goto LABEL_46;
        v56 = *(_DWORD *)(v50 + 32);
        v57 = 0;
        v58 = 0;
        v74 = v56;
        while ( v74 != (_DWORD)v57 )
        {
          v62 = v17;
          v64 = v53;
          v59 = sub_AD69F0(v53, v57);
          v53 = v64;
          v17 = v62;
          if ( !v59 )
            goto LABEL_46;
          if ( *(_BYTE *)v59 != 13 )
          {
            if ( *(_BYTE *)v59 != 17 )
              goto LABEL_46;
            v60 = *(_DWORD *)(v59 + 32);
            if ( v60 <= 0x40 )
            {
              v58 = *(_QWORD *)(v59 + 24) == 0;
            }
            else
            {
              v61 = sub_C444A0(v59 + 24);
              v17 = v62;
              v53 = v64;
              v58 = v60 == v61;
            }
            if ( !v58 )
              goto LABEL_46;
          }
          v57 = (unsigned int)(v57 + 1);
        }
        if ( !v58 )
          goto LABEL_46;
LABEL_70:
        if ( v18 != *((_QWORD *)v34 - 4) )
          goto LABEL_46;
        v39 = *(_BYTE **)(a3 - 32);
        if ( !v39 )
          goto LABEL_77;
        if ( *v39 != 17 )
        {
          v47 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v39 + 1) + 8LL) - 17;
          if ( (unsigned int)v47 > 1 )
            goto LABEL_46;
          if ( *v39 > 0x15u )
            goto LABEL_46;
          v71 = v17;
          v48 = sub_AD7630(*(_QWORD *)(a3 - 32), 0, v47);
          v17 = v71;
          v39 = v48;
          if ( !v48 || *v48 != 17 )
            goto LABEL_46;
        }
        v40 = *((_DWORD *)v39 + 8);
        if ( v40 <= 0x40 )
        {
          v41 = *((_QWORD *)v39 + 3);
          goto LABEL_75;
        }
        v72 = v17;
        v49 = sub_C444A0((__int64)(v39 + 24));
        v17 = v72;
        if ( v40 - v49 <= 0x40 )
        {
          v41 = **((_QWORD **)v39 + 3);
LABEL_75:
          if ( v17 == v41 )
            goto LABEL_15;
        }
LABEL_46:
        v16 = *a2;
        goto LABEL_13;
      }
      v54 = *((_DWORD *)v52 + 8);
      if ( v54 <= 0x40 )
      {
        v38 = *((_QWORD *)v52 + 3) == 0;
      }
      else
      {
        v55 = sub_C444A0((__int64)(v52 + 24));
        v17 = v63;
        v38 = v54 == v55;
      }
    }
    if ( !v38 )
      goto LABEL_46;
    goto LABEL_70;
  }
LABEL_13:
  if ( v16 != 68 )
    goto LABEL_14;
  v27 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v27 != 57 )
    goto LABEL_14;
  v18 = *((_QWORD *)v27 - 8);
  if ( !v18 )
    goto LABEL_14;
  v28 = *((_QWORD *)v27 - 4);
  if ( !v28 )
    goto LABEL_77;
  if ( *(_BYTE *)v28 != 17 )
  {
    v68 = v17;
    v42 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v28 + 8) + 8LL) - 17;
    if ( (unsigned int)v42 > 1 )
      goto LABEL_14;
    if ( *(_BYTE *)v28 > 0x15u )
      goto LABEL_14;
    v43 = sub_AD7630(v28, 0, v42);
    v28 = (__int64)v43;
    if ( !v43 )
      goto LABEL_14;
    v17 = v68;
    if ( *v43 != 17 )
      goto LABEL_14;
  }
  v29 = *(_DWORD *)(v28 + 32);
  if ( v29 > 0x40 )
  {
    v69 = v17;
    v46 = sub_C444A0(v28 + 24);
    v17 = v69;
    if ( v29 - v46 <= 0x40 )
    {
      v30 = **(_QWORD **)(v28 + 24);
      goto LABEL_54;
    }
LABEL_14:
    v18 = 0;
    goto LABEL_15;
  }
  v30 = *(_QWORD *)(v28 + 24);
LABEL_54:
  if ( v17 != v30 )
    goto LABEL_14;
  v31 = *(_BYTE *)a3 == 68;
  v77.m128i_i64[0] = 0;
  v77.m128i_i64[1] = v18;
  v78.m128i_i64[0] = v17;
  if ( !v31 || !sub_10C9A60((__int64)&v77, 28, *(unsigned __int8 **)(a3 - 32)) )
    goto LABEL_14;
LABEL_15:
  if ( v76 > 0x40 && v75 )
    j_j___libc_free_0_0(v75);
  return v18;
}
