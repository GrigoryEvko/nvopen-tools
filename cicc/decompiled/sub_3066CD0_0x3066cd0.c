// Function: sub_3066CD0
// Address: 0x3066cd0
//
__int64 __fastcall sub_3066CD0(
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
  __int64 v12; // rbx
  __int64 v13; // rcx
  unsigned __int16 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r12
  __int64 v18; // r9
  __int64 v19; // r15
  int v20; // r13d
  int v21; // edx
  __int64 v23; // rax
  int v24; // eax
  int v25; // edx
  signed __int64 v26; // rcx
  signed __int64 v27; // r13
  unsigned __int64 v28; // r12
  unsigned int v29; // eax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r12
  bool v32; // of
  bool v33; // cc
  __int64 *v34; // [rsp+8h] [rbp-88h]
  signed __int64 v36; // [rsp+28h] [rbp-68h]
  unsigned int v38; // [rsp+3Ch] [rbp-54h]
  unsigned int v39; // [rsp+3Ch] [rbp-54h]
  unsigned __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 24);
  v38 = sub_2FEBEF0(v11, a2);
  if ( a6 )
    return 1;
  if ( v38 == 205 )
  {
    v24 = 206;
    if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 > 1 )
      v24 = 205;
    v38 = v24;
  }
  v36 = 1;
  v12 = *a3;
  v13 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, 0);
  v14 = v13;
  v34 = a3;
  v16 = v15;
  while ( 1 )
  {
    LOWORD(v13) = v14;
    sub_2FE6CC0((__int64)&v40, *(_QWORD *)(a1 + 24), v12, v13, v16);
    if ( (_BYTE)v40 == 10 )
      break;
    if ( !(_BYTE)v40 )
      goto LABEL_9;
    if ( (v40 & 0xFB) == 2 )
    {
      v23 = 2 * v36;
      if ( !is_mul_ok(2u, v36) )
      {
        v23 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v36 <= 0 )
          v23 = 0x8000000000000000LL;
      }
      v36 = v23;
    }
    if ( (_WORD)v41 == v14 && ((_WORD)v41 || v42 == v16) )
    {
LABEL_9:
      v17 = (__int64)v34;
      v18 = v14;
      v19 = a1;
      v20 = a2;
      v21 = *((unsigned __int8 *)v34 + 8);
      if ( (unsigned int)(v21 - 17) > 1 )
      {
        if ( !(_WORD)v18 )
          goto LABEL_18;
      }
      else if ( (unsigned __int16)(v18 - 17) > 0xD3u )
      {
        goto LABEL_18;
      }
      goto LABEL_11;
    }
    v13 = v41;
    v16 = v42;
    v14 = v41;
  }
  v17 = (__int64)v34;
  v18 = v14;
  v19 = a1;
  v20 = a2;
  v21 = *((unsigned __int8 *)v34 + 8);
  if ( !(_WORD)v18 )
    v18 = 8;
  if ( (unsigned int)(v21 - 17) <= 1 && (unsigned __int16)(v18 - 17) > 0xD3u )
    goto LABEL_18;
  v36 = 0;
LABEL_11:
  if ( *(_QWORD *)(v11 + 8 * v18 + 112)
    && (v38 > 0x1F3 || *(_BYTE *)(v38 + 500LL * (unsigned __int16)v18 + v11 + 6414) != 2) )
  {
    return v36;
  }
LABEL_18:
  if ( v21 == 17 )
  {
    v39 = *(_DWORD *)(v17 + 32);
    if ( a4 && (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
      a4 = **(_QWORD **)(a4 + 16);
    v26 = sub_3066CD0(v19, v20, **(_QWORD **)(v17 + 16), a4, a5, 0, a7, a8, a9);
    if ( is_mul_ok(v26, v39) )
    {
      v27 = v26 * v39;
    }
    else if ( !v39 || (v27 = 0x7FFFFFFFFFFFFFFFLL, v26 <= 0) )
    {
      v27 = 0x8000000000000000LL;
    }
    if ( *(_BYTE *)(v17 + 8) == 18 )
    {
      if ( v25 == 1 )
        return v27;
      return v27;
    }
    else
    {
      v29 = *(_DWORD *)(v17 + 32);
      LODWORD(v41) = v29;
      if ( v29 > 0x40 )
      {
        sub_C43690((__int64)&v40, -1, 1);
      }
      else
      {
        v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
        if ( !v29 )
          v30 = 0;
        v40 = v30;
      }
      v31 = sub_3064F80(v19, v17, (__int64 *)&v40, 1, 0);
      if ( (unsigned int)v41 > 0x40 && v40 )
        j_j___libc_free_0_0(v40);
      v32 = __OFADD__(v27, v31);
      v28 = v27 + v31;
      if ( v32 )
      {
        v33 = v27 <= 0;
        v27 = 0x8000000000000000LL;
        if ( !v33 )
          return 0x7FFFFFFFFFFFFFFFLL;
        return v27;
      }
    }
    return v28;
  }
  return v21 != 18;
}
