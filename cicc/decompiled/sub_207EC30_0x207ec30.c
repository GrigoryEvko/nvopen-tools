// Function: sub_207EC30
// Address: 0x207ec30
//
__int64 __fastcall sub_207EC30(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  unsigned int v8; // r15d
  unsigned __int64 v11; // r12
  unsigned int v12; // ebx
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // edx
  __int64 result; // rax
  __int64 v25; // r12
  unsigned int v26; // r14d
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // r9d
  int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 *v33; // r11
  __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned int v39; // edx
  __int64 v40; // rsi
  __int64 v41; // r8
  unsigned int v42; // edx
  __int64 v43; // r9
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned int v46; // edx
  char v47; // al
  unsigned int v48; // ecx
  unsigned int v49; // edx
  __int64 v50; // r8
  __int64 *v51; // r11
  __int64 *v52; // [rsp+8h] [rbp-68h]
  __int64 *v53; // [rsp+10h] [rbp-60h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  __int64 v55; // [rsp+10h] [rbp-60h]
  __int64 v56; // [rsp+10h] [rbp-60h]
  unsigned int v57; // [rsp+18h] [rbp-58h]
  int v58; // [rsp+24h] [rbp-4Ch]
  unsigned int v59; // [rsp+28h] [rbp-48h]
  __int64 *v60; // [rsp+28h] [rbp-48h]
  _QWORD *v61; // [rsp+28h] [rbp-48h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 *v64; // [rsp+28h] [rbp-48h]
  __int64 v66; // [rsp+38h] [rbp-38h]
  int v67; // [rsp+38h] [rbp-38h]
  __int64 v68; // [rsp+38h] [rbp-38h]
  int v69; // [rsp+38h] [rbp-38h]
  unsigned __int64 v70; // [rsp+38h] [rbp-38h]

  v11 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = a2;
  v13 = *(_BYTE *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (a1 & 4) != 0 )
  {
    if ( v13 < 0 )
    {
      v14 = sub_1648A40(v11);
      v66 = v15 + v14;
      if ( *(char *)(v11 + 23) >= 0 )
      {
        if ( (unsigned int)(v66 >> 4) )
          goto LABEL_49;
      }
      else if ( (unsigned int)((v66 - sub_1648A40(v11)) >> 4) )
      {
        if ( *(char *)(v11 + 23) < 0 )
        {
          v67 = *(_DWORD *)(sub_1648A40(v11) + 8);
          if ( *(char *)(v11 + 23) >= 0 )
            BUG();
          v16 = sub_1648A40(v11);
          v18 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v67);
          goto LABEL_16;
        }
LABEL_49:
        BUG();
      }
    }
    v18 = -24;
    goto LABEL_16;
  }
  if ( v13 >= 0 )
    goto LABEL_15;
  v19 = sub_1648A40(v11);
  v68 = v20 + v19;
  if ( *(char *)(v11 + 23) >= 0 )
  {
    if ( (unsigned int)(v68 >> 4) )
LABEL_51:
      BUG();
LABEL_15:
    v18 = -72;
    goto LABEL_16;
  }
  if ( !(unsigned int)((v68 - sub_1648A40(v11)) >> 4) )
    goto LABEL_15;
  if ( *(char *)(v11 + 23) >= 0 )
    goto LABEL_51;
  v69 = *(_DWORD *)(sub_1648A40(v11) + 8);
  if ( *(char *)(v11 + 23) >= 0 )
    BUG();
  v21 = sub_1648A40(v11);
  v18 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v69);
LABEL_16:
  v23 = *(_DWORD *)(v11 + 20);
  result = 0xAAAAAAAAAAAAAAABLL * ((v18 + 24LL * (v23 & 0xFFFFFFF)) >> 3);
  v58 = result;
  if ( a2 != (_DWORD)result )
  {
    v70 = a1 & 0xFFFFFFFFFFFFFFF8LL;
    v25 = a4;
    v26 = v59;
    while ( 1 )
    {
      v51 = sub_20685E0(a5, *(__int64 **)(v70 + 24 * (v12 - (unsigned __int64)(v23 & 0xFFFFFFF))), a6, a7, a8);
      v30 = *((unsigned __int16 *)v51 + 12);
      v50 = v28;
      if ( v30 == 32 || v30 == 10 )
        break;
      if ( v30 == 36 || v30 == 14 )
      {
        v53 = v51;
        v61 = *(_QWORD **)(a5 + 552);
        v45 = sub_1E0A0C0(v61[4]);
        v46 = 8 * sub_15A9520(v45, *(_DWORD *)(v45 + 4));
        if ( v46 == 32 )
        {
          v47 = 5;
        }
        else if ( v46 > 0x20 )
        {
          v47 = 6;
          if ( v46 != 64 )
          {
            v47 = 0;
            if ( v46 == 128 )
              v47 = 7;
          }
        }
        else
        {
          v47 = 3;
          if ( v46 != 8 )
            v47 = 4 * (v46 == 16);
        }
        v48 = v57;
        LOBYTE(v48) = v47;
        v41 = (__int64)sub_1D299D0(v61, *((_DWORD *)v53 + 21), v48, 0, 1);
        v43 = v49;
        v44 = *(unsigned int *)(v25 + 8);
        if ( (unsigned int)v44 >= *(_DWORD *)(v25 + 12) )
          goto LABEL_41;
        goto LABEL_31;
      }
      v27 = *(unsigned int *)(v25 + 8);
      if ( (unsigned int)v27 >= *(_DWORD *)(v25 + 12) )
      {
        v56 = v28;
        v64 = v51;
        sub_16CD150(v25, (const void *)(v25 + 16), 0, 16, v28, v29);
        v27 = *(unsigned int *)(v25 + 8);
        v50 = v56;
        v51 = v64;
      }
      result = *(_QWORD *)v25 + 16 * v27;
      ++v12;
      *(_QWORD *)result = v51;
      *(_QWORD *)(result + 8) = v50;
      ++*(_DWORD *)(v25 + 8);
      if ( v12 == v58 )
        return result;
LABEL_24:
      v23 = *(_DWORD *)(v70 + 20);
    }
    LOBYTE(v26) = 6;
    v60 = v51;
    v31 = sub_1D38BB0(*(_QWORD *)(a5 + 552), 2, a3, v26, 0, 1, a6, *(double *)a7.m128i_i64, a8, 0);
    v33 = v60;
    v34 = v31;
    v35 = *(unsigned int *)(v25 + 8);
    v36 = v32;
    if ( (unsigned int)v35 >= *(_DWORD *)(v25 + 12) )
    {
      v52 = v60;
      v55 = v34;
      v63 = v32;
      sub_16CD150(v25, (const void *)(v25 + 16), 0, 16, v34, v32);
      v35 = *(unsigned int *)(v25 + 8);
      v33 = v52;
      v34 = v55;
      v36 = v63;
    }
    v37 = (__int64 *)(*(_QWORD *)v25 + 16 * v35);
    *v37 = v34;
    v37[1] = v36;
    ++*(_DWORD *)(v25 + 8);
    v38 = v33[11];
    v39 = *(_DWORD *)(v38 + 32);
    if ( v39 > 0x40 )
      v40 = **(_QWORD **)(v38 + 24);
    else
      v40 = (__int64)(*(_QWORD *)(v38 + 24) << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
    LOBYTE(v8) = 6;
    v41 = sub_1D38BB0(*(_QWORD *)(a5 + 552), v40, a3, v8, 0, 1, a6, *(double *)a7.m128i_i64, a8, 0);
    v43 = v42;
    v44 = *(unsigned int *)(v25 + 8);
    if ( (unsigned int)v44 >= *(_DWORD *)(v25 + 12) )
    {
LABEL_41:
      v54 = v41;
      v62 = v43;
      sub_16CD150(v25, (const void *)(v25 + 16), 0, 16, v41, v43);
      v44 = *(unsigned int *)(v25 + 8);
      v41 = v54;
      v43 = v62;
    }
LABEL_31:
    result = *(_QWORD *)v25 + 16 * v44;
    ++v12;
    *(_QWORD *)result = v41;
    *(_QWORD *)(result + 8) = v43;
    ++*(_DWORD *)(v25 + 8);
    if ( v12 == v58 )
      return result;
    goto LABEL_24;
  }
  return result;
}
