// Function: sub_2B6FB20
// Address: 0x2b6fb20
//
__int64 __fastcall sub_2B6FB20(
        unsigned __int8 **a1,
        __int64 (__fastcall *a2)(__int64, _QWORD, __int64),
        __int64 a3,
        __int64 (__fastcall *a4)(__int64, unsigned __int8 *, unsigned __int8 *),
        __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  signed __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // r11
  unsigned __int64 v15; // r13
  unsigned __int8 *v16; // r9
  int v17; // edx
  __int64 v18; // r8
  _BYTE *v19; // rax
  __int64 v20; // r11
  __int64 **v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // r13d
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 result; // rax
  __int64 v39; // r12
  __int64 *v40; // r15
  __int64 v41; // rbx
  __int64 v42; // rdi
  unsigned int v44; // r15d
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  bool v47; // of
  unsigned __int64 v48; // rdx
  __int64 v49; // [rsp+0h] [rbp-80h]
  __int64 v50; // [rsp+0h] [rbp-80h]
  unsigned int v52; // [rsp+8h] [rbp-78h]
  __int64 v54; // [rsp+10h] [rbp-70h]
  __int64 v55; // [rsp+10h] [rbp-70h]
  signed __int64 v56; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v57; // [rsp+18h] [rbp-68h]
  __int64 v58; // [rsp+18h] [rbp-68h]
  unsigned int v59; // [rsp+18h] [rbp-68h]
  __int64 v60; // [rsp+18h] [rbp-68h]
  __int64 v61; // [rsp+20h] [rbp-60h] BYREF
  __int64 v62; // [rsp+28h] [rbp-58h]
  __int64 *v63; // [rsp+30h] [rbp-50h] BYREF
  __int64 v64; // [rsp+40h] [rbp-40h]

  v7 = **a1;
  if ( (unsigned int)(v7 - 67) <= 0xC || (_BYTE)v7 == 85 )
  {
    v8 = a2(a3, 0, v7);
    v9 = (unsigned __int64)a1[2];
    v10 = *((unsigned int *)a1 + 2);
    v56 = v8;
    if ( (v9 & 1) != 0 )
    {
      v10 -= (int)sub_39FAC40(~(-1LL << (v9 >> 58)) & (v9 >> 1));
    }
    else
    {
      v39 = *(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v39 )
      {
        v40 = *(__int64 **)v9;
        LODWORD(v41) = 0;
        do
        {
          v42 = *v40++;
          v41 = (unsigned int)sub_39FAC40(v42) + (unsigned int)v41;
        }
        while ( (__int64 *)v39 != v40 );
        v10 -= v41;
      }
    }
    v11 = v10 * v56;
    if ( !is_mul_ok(v10, v56) )
    {
      if ( v10 <= 0 )
      {
        if ( v10 >= 0 || (v11 = 0x7FFFFFFFFFFFFFFFLL, v56 >= 0) )
          v11 = 0x8000000000000000LL;
      }
      else
      {
        v11 = 0x8000000000000000LL;
        if ( v56 > 0 )
          v11 = 0x7FFFFFFFFFFFFFFFLL;
      }
    }
  }
  else if ( *((_DWORD *)a1 + 2) )
  {
    v44 = 0;
    v11 = 0;
    do
    {
      v48 = (unsigned __int64)a1[2];
      if ( (v48 & 1) != 0 )
        v45 = (((v48 >> 1) & ~(-1LL << ((unsigned __int64)a1[2] >> 58))) >> v44) & 1;
      else
        v45 = (*(_QWORD *)(*(_QWORD *)v48 + 8LL * (v44 >> 6)) >> v44) & 1LL;
      if ( !(_BYTE)v45 )
      {
        v46 = ((__int64 (__fastcall *)(__int64, _QWORD))a2)(a3, v44);
        v47 = __OFADD__(v46, v11);
        v11 += v46;
        if ( v47 )
        {
          v11 = 0x8000000000000000LL;
          if ( v46 > 0 )
            v11 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
      ++v44;
    }
    while ( *((_DWORD *)a1 + 2) > v44 );
  }
  else
  {
    v11 = 0;
  }
  v12 = a4(a5, a1[3], a1[4]);
  v13 = a1[9];
  v14 = v12;
  v15 = v12;
  if ( a1[7] != (unsigned __int8 *)(*((_QWORD *)v13 + 441) + 24LL * *((unsigned int *)v13 + 886)) )
  {
    v16 = a1[10];
    v17 = **((unsigned __int8 **)v16 + 52);
    if ( (unsigned int)(v17 - 67) > 0xC )
    {
      if ( *((_DWORD *)v16 + 50) )
      {
        v18 = *((_QWORD *)v16 + 23);
        if ( (_BYTE)v17 != 61 || v18 )
        {
          v19 = *(_BYTE **)(v18 + 416);
          if ( !v19 || !*(_QWORD *)(v18 + 424) || *v19 != 86 || *((_DWORD *)v16 + 48) )
          {
            v54 = v14;
            v57 = a1[10];
            sub_2B3B8A0(&v63, (__int64 *)v13 + 440, *((_QWORD *)v16 + 23));
            v20 = v54;
            v21 = (__int64 **)*((_QWORD *)v57 + 23);
            if ( ((*((_DWORD *)v21 + 26) - 3) & 0xFFFFFFFD) != 0 )
              v22 = *(_QWORD *)v21[30][10 * *((unsigned int *)v57 + 48)];
            else
              v22 = **v21;
            v23 = *(_QWORD *)(v22 + 8);
            if ( v64 != *((_QWORD *)a1[9] + 441) + 24LL * *((unsigned int *)a1[9] + 886) )
            {
              v24 = sub_BCCE00(*(_QWORD **)a1[11], *(_QWORD *)(v64 + 8));
              v20 = v54;
              v23 = v24;
            }
            v25 = (__int64)a1[11];
            v55 = v20;
            v58 = v23;
            if ( v25 != v23 )
            {
              v26 = sub_9208B0(*((_QWORD *)a1[9] + 418), v25);
              v62 = v27;
              v61 = v26;
              v52 = sub_CA1930(&v61);
              v49 = v58;
              v28 = sub_9208B0(*((_QWORD *)a1[9] + 418), v58);
              v29 = 38;
              v62 = v30;
              v61 = v28;
              v59 = sub_CA1930(&v61);
              v50 = sub_2B08680(v49, *((_DWORD *)a1[10] + 2));
              if ( v52 <= v59 )
                v29 = 39 - ((a1[7][16] == 0) - 1);
              v60 = (__int64)a1[12];
              v31 = (__int64 *)sub_2B2A0E0(v60, (__int64)*a1);
              if ( v35 == 1 )
              {
                sub_2B2D870(v60, *v31, 1, v32, v33, v34);
              }
              else
              {
                v36 = *(_QWORD *)(*(_QWORD *)a1[13] + 240LL);
                sub_2B5F980(*(__int64 **)v36, *(unsigned int *)(v36 + 8), *(__int64 **)(v60 + 3304));
              }
              v37 = sub_DFD060(*((__int64 **)a1[9] + 412), v29, v50, (__int64)a1[14]);
              if ( __OFADD__(v37, v55) )
              {
                v15 = 0x7FFFFFFFFFFFFFFFLL;
                if ( v37 <= 0 )
                  v15 = 0x8000000000000000LL;
              }
              else
              {
                v15 = v37 + v55;
              }
            }
          }
        }
      }
    }
  }
  result = v15 - v11;
  if ( __OFSUB__(v15, v11) )
  {
    result = 0x8000000000000000LL;
    if ( v11 <= 0 )
      return 0x7FFFFFFFFFFFFFFFLL;
  }
  return result;
}
