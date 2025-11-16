// Function: sub_34D1290
// Address: 0x34d1290
//
__int64 __fastcall sub_34D1290(
        __int64 a1,
        int a2,
        __int64 *a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rbx
  unsigned __int16 i; // r15
  __int64 *v18; // rbx
  __int64 v19; // r12
  int v20; // r13d
  __int64 v21; // r9
  int v22; // ecx
  __int64 result; // rax
  __int64 v24; // rax
  int v25; // eax
  unsigned __int64 v26; // r14
  signed __int64 v27; // rax
  signed __int64 v28; // r15
  int v29; // edx
  int v30; // r13d
  signed __int64 v31; // rdx
  int v32; // r8d
  unsigned int v33; // edi
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // r14
  __int64 *v36; // r15
  unsigned int v37; // ebx
  __int64 v38; // r13
  int v39; // r12d
  unsigned __int64 v40; // rax
  __int64 *v41; // rsi
  unsigned int v42; // eax
  bool v43; // of
  bool v44; // cc
  __int64 *v45; // [rsp+8h] [rbp-88h]
  signed __int64 v47; // [rsp+28h] [rbp-68h]
  unsigned int v49; // [rsp+38h] [rbp-58h]
  signed __int64 v50; // [rsp+38h] [rbp-58h]
  unsigned __int64 v51; // [rsp+40h] [rbp-50h] BYREF
  __int64 v52; // [rsp+48h] [rbp-48h]
  __int64 v53; // [rsp+50h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 24);
  v49 = sub_2FEBEF0(v11, a2);
  if ( !a6 )
  {
    if ( v49 == 205 )
    {
      v25 = 206;
      if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 > 1 )
        v25 = 205;
      v49 = v25;
    }
    v12 = *a3;
    v47 = 1;
    v13 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, 0);
    v15 = v14;
    v45 = a3;
    v16 = v12;
    for ( i = v13; ; i = v52 )
    {
      LOWORD(v13) = i;
      sub_2FE6CC0((__int64)&v51, *(_QWORD *)(a1 + 24), v16, v13, v15);
      if ( (_BYTE)v51 == 10 )
        break;
      if ( !(_BYTE)v51 )
        goto LABEL_9;
      if ( (v51 & 0xFB) == 2 )
      {
        v24 = 2 * v47;
        if ( !is_mul_ok(2u, v47) )
        {
          v24 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v47 <= 0 )
            v24 = 0x8000000000000000LL;
        }
        v47 = v24;
      }
      if ( (_WORD)v52 == i && ((_WORD)v52 || v15 == v53) )
      {
LABEL_9:
        v18 = v45;
        v19 = a1;
        v20 = a2;
        v21 = i;
        v22 = *((unsigned __int8 *)v45 + 8);
        if ( (unsigned int)(v22 - 17) > 1 )
        {
          if ( !i )
            goto LABEL_18;
        }
        else if ( (unsigned __int16)(i - 17) > 0xD3u )
        {
          goto LABEL_18;
        }
        goto LABEL_11;
      }
      v13 = v52;
      v15 = v53;
    }
    v21 = i;
    v19 = a1;
    v18 = v45;
    v20 = a2;
    if ( !i )
      v21 = 8;
    v22 = *((unsigned __int8 *)v45 + 8);
    if ( (unsigned int)(v22 - 17) <= 1 && (unsigned __int16)(v21 - 17) > 0xD3u )
      goto LABEL_18;
    v47 = 0;
LABEL_11:
    if ( *(_QWORD *)(v11 + 8 * v21 + 112)
      && (v49 > 0x1F3 || *(_BYTE *)(v49 + 500LL * (unsigned __int16)v21 + v11 + 6414) != 2) )
    {
      return v47;
    }
LABEL_18:
    if ( v22 != 17 )
    {
      result = 0;
      if ( v22 == 18 )
        return result;
      return 1;
    }
    v26 = *((unsigned int *)v18 + 8);
    if ( a4 && (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
      a4 = **(_QWORD **)(a4 + 16);
    v27 = sub_34D1290(v19, v20, *(_QWORD *)v18[2], a4, a5, 0, a7, a8, a9);
    v28 = v27 * v26;
    v30 = v29;
    if ( !is_mul_ok(v27, v26) )
    {
      if ( !v26 || (v28 = 0x7FFFFFFFFFFFFFFFLL, v27 <= 0) )
        v28 = 0x8000000000000000LL;
    }
    if ( *((_BYTE *)v18 + 8) == 18 )
    {
      v31 = v28;
      if ( v30 == 1 )
        return v28;
      return v31;
    }
    v32 = *((_DWORD *)v18 + 8);
    LODWORD(v52) = v32;
    if ( (unsigned int)v32 > 0x40 )
    {
      sub_C43690((__int64)&v51, -1, 1);
      if ( *((_BYTE *)v18 + 8) == 18 )
      {
        v33 = v52;
        v35 = 0;
        goto LABEL_49;
      }
      v32 = *((_DWORD *)v18 + 8);
      v33 = v52;
    }
    else
    {
      v33 = v32;
      v34 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v32;
      if ( !v32 )
        v34 = 0;
      v51 = v34;
    }
    v35 = 0;
    if ( v32 > 0 )
    {
      v50 = v28;
      v36 = v18;
      v37 = 0;
      v38 = v19;
      v39 = v32;
      do
      {
        v40 = v51;
        if ( v33 > 0x40 )
          v40 = *(_QWORD *)(v51 + 8LL * (v37 >> 6));
        if ( (v40 & (1LL << v37)) != 0 )
        {
          v41 = v36;
          if ( (unsigned int)*((unsigned __int8 *)v36 + 8) - 17 <= 1 )
            v41 = *(__int64 **)v36[2];
          v42 = sub_34D06B0(v38, v41);
          v33 = v52;
          v43 = __OFADD__(v42, v35);
          v35 += v42;
          if ( v43 )
          {
            v35 = 0x8000000000000000LL;
            if ( v42 )
              v35 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
        ++v37;
      }
      while ( v39 != v37 );
      v28 = v50;
    }
LABEL_49:
    if ( v33 > 0x40 && v51 )
      j_j___libc_free_0_0(v51);
    v31 = v28 + v35;
    if ( __OFADD__(v28, v35) )
    {
      v44 = v28 <= 0;
      v28 = 0x8000000000000000LL;
      if ( !v44 )
        return 0x7FFFFFFFFFFFFFFFLL;
      return v28;
    }
    return v31;
  }
  return 1;
}
