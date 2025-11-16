// Function: sub_22C7100
// Address: 0x22c7100
//
__int64 __fastcall sub_22C7100(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v7; // al
  __int8 v9; // al
  int v10; // eax
  int v11; // eax
  __int64 v12; // rax
  __int64 v14; // rdi
  unsigned int v16; // esi
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // r8
  int v22; // r9d
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 *v27; // r9
  bool v28; // zf
  int v29; // eax
  char v30; // al
  __int64 v31; // rdi
  int v32; // r8d
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // r9
  int v36; // eax
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  int v39; // edx
  int v40; // r10d
  int v41; // esi
  __int64 v42; // rdi
  unsigned __int64 v43; // rsi
  int v44; // r10d
  int v45; // esi
  int v46; // r10d
  __int64 v47; // [rsp+8h] [rbp-B8h]
  __int64 v48; // [rsp+10h] [rbp-B0h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  __int64 v50; // [rsp+10h] [rbp-B0h]
  __int64 v51; // [rsp+10h] [rbp-B0h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+18h] [rbp-A8h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  __int64 v59; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+28h] [rbp-98h]
  __int64 v61[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v62[4]; // [rsp+40h] [rbp-80h] BYREF
  __m128i v63; // [rsp+60h] [rbp-60h] BYREF
  __int64 v64; // [rsp+70h] [rbp-50h]
  __int64 v65; // [rsp+78h] [rbp-48h]
  int v66; // [rsp+80h] [rbp-40h]
  char v67; // [rsp+88h] [rbp-38h]

  v7 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    v63.m128i_i16[0] = 0;
    if ( (unsigned __int8)(v7 - 12) <= 1u )
    {
      *(_WORD *)a1 = 1;
      goto LABEL_4;
    }
    if ( v7 == 17 )
    {
      v60 = *(_DWORD *)(a3 + 32);
      if ( v60 > 0x40 )
        sub_C43780((__int64)&v59, (const void **)(a3 + 24));
      else
        v59 = *(_QWORD *)(a3 + 24);
      sub_AADBC0((__int64)v61, &v59);
      sub_22C00F0((__int64)&v63, (__int64)v61, 0, 0, 1u);
      sub_969240(v62);
      sub_969240(v61);
      sub_969240(&v59);
      v9 = v63.m128i_i8[0];
      *(_WORD *)a1 = v63.m128i_u8[0];
      if ( (unsigned __int8)v9 > 3u )
      {
        if ( (unsigned __int8)(v9 - 4) <= 1u )
        {
          v10 = v64;
          LODWORD(v64) = 0;
          *(_DWORD *)(a1 + 16) = v10;
          *(_QWORD *)(a1 + 8) = v63.m128i_i64[1];
          v11 = v66;
          v66 = 0;
          *(_DWORD *)(a1 + 32) = v11;
          *(_QWORD *)(a1 + 24) = v65;
          *(_BYTE *)(a1 + 1) = v63.m128i_i8[1];
        }
        goto LABEL_4;
      }
      if ( (unsigned __int8)v9 <= 1u )
      {
LABEL_4:
        *(_BYTE *)(a1 + 40) = 1;
        v63.m128i_i8[0] = 0;
        sub_22C0090((unsigned __int8 *)&v63);
        return a1;
      }
    }
    else
    {
      v63.m128i_i64[1] = a3;
      *(_WORD *)a1 = 2;
    }
    *(_QWORD *)(a1 + 8) = v63.m128i_i64[1];
    goto LABEL_4;
  }
  v12 = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)v12 )
    goto LABEL_30;
  v14 = *(_QWORD *)(a2 + 8);
  v16 = (v12 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v17 = v14 + 48LL * v16;
  v18 = *(_QWORD *)(v17 + 24);
  if ( v18 != a4 )
  {
    v39 = 1;
    while ( v18 != -4096 )
    {
      v40 = v39 + 1;
      v16 = (v12 - 1) & (v39 + v16);
      v17 = v14 + 48LL * v16;
      v18 = *(_QWORD *)(v17 + 24);
      if ( a4 == v18 )
        goto LABEL_14;
      v39 = v40;
    }
    goto LABEL_30;
  }
LABEL_14:
  if ( v17 == v14 + 48 * v12 || (v19 = *(_QWORD *)(v17 + 40)) == 0 )
  {
LABEL_30:
    v63.m128i_i64[0] = a4;
    v63.m128i_i64[1] = a3;
    if ( (unsigned __int8)sub_22C3FB0(a2, &v63) )
    {
      *(_BYTE *)(a1 + 40) = 0;
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 1;
      *(_WORD *)a1 = 6;
      v63.m128i_i16[0] = 0;
      sub_22C0090((unsigned __int8 *)&v63);
    }
    return a1;
  }
  v61[0] = 0;
  v61[1] = 0;
  v62[0] = a3;
  if ( a3 == -4096 || a3 == -8192 )
  {
    v20 = a3;
  }
  else
  {
    v48 = a4;
    v52 = v19;
    sub_BD73F0((__int64)v61);
    v20 = v62[0];
    v19 = v52;
    a4 = v48;
  }
  if ( (*(_BYTE *)(v19 + 280) & 1) != 0 )
  {
    v21 = v19 + 288;
    v22 = 3;
  }
  else
  {
    v29 = *(_DWORD *)(v19 + 296);
    v21 = *(_QWORD *)(v19 + 288);
    v22 = v29 - 1;
    if ( !v29 )
      goto LABEL_37;
  }
  v63 = 0u;
  v64 = -4096;
  v23 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
  v24 = *(_QWORD *)(v21 + 24LL * v23 + 16);
  if ( v24 == v20 )
  {
LABEL_22:
    v53 = a4;
    sub_D68D70(&v63);
    v25 = v53;
    if ( v62[0] && v62[0] != -8192 && v62[0] != -4096 )
    {
      sub_BD60C0(v61);
      v25 = v53;
    }
    v54 = v25;
    LOWORD(v61[0]) = 0;
    v63.m128i_i64[0] = 6;
    v67 = 1;
    sub_22C0090((unsigned __int8 *)v61);
    a4 = v54;
    if ( v67 )
      goto LABEL_27;
    goto LABEL_30;
  }
  v44 = 1;
  while ( v24 != -4096 )
  {
    v23 = v22 & (v44 + v23);
    v24 = *(_QWORD *)(v21 + 24LL * v23 + 16);
    if ( v24 == v20 )
      goto LABEL_22;
    ++v44;
  }
  v50 = a4;
  v57 = v19;
  sub_D68D70(&v63);
  v20 = v62[0];
  v19 = v57;
  a4 = v50;
LABEL_37:
  if ( v20 && v20 != -8192 && v20 != -4096 )
  {
    v49 = a4;
    v55 = v19;
    sub_BD60C0(v61);
    v19 = v55;
    a4 = v49;
  }
  v30 = *(_BYTE *)(v19 + 8);
  if ( (v30 & 1) != 0 )
  {
    v31 = v19 + 16;
    v32 = 3;
  }
  else
  {
    v41 = *(_DWORD *)(v19 + 24);
    v31 = *(_QWORD *)(v19 + 16);
    v32 = v41 - 1;
    if ( !v41 )
      goto LABEL_56;
  }
  v63 = 0u;
  v64 = -4096;
  v33 = v32 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v34 = v31 + ((unsigned __int64)v33 << 6);
  v35 = *(_QWORD *)(v34 + 16);
  if ( a3 == v35 )
  {
LABEL_44:
    v47 = a4;
    v56 = v19;
    sub_D68D70(&v63);
    v19 = v56;
    a4 = v47;
    LOBYTE(v36) = *(_BYTE *)(v56 + 8) & 1;
    goto LABEL_45;
  }
  v45 = 1;
  while ( v35 != -4096 )
  {
    v46 = v45 + 1;
    v33 = v32 & (v45 + v33);
    v34 = v31 + ((unsigned __int64)v33 << 6);
    v35 = *(_QWORD *)(v34 + 16);
    if ( a3 == v35 )
      goto LABEL_44;
    v45 = v46;
  }
  v51 = a4;
  v58 = v19;
  sub_D68D70(&v63);
  v19 = v58;
  a4 = v51;
  v30 = *(_BYTE *)(v58 + 8);
LABEL_56:
  v36 = v30 & 1;
  if ( v36 )
  {
    v42 = v19 + 16;
    v43 = 256;
  }
  else
  {
    v42 = *(_QWORD *)(v19 + 16);
    v43 = (unsigned __int64)*(unsigned int *)(v19 + 24) << 6;
  }
  v34 = v42 + v43;
LABEL_45:
  if ( (_BYTE)v36 )
  {
    v37 = v19 + 16;
    v38 = 256;
  }
  else
  {
    v37 = *(_QWORD *)(v19 + 16);
    v38 = (unsigned __int64)*(unsigned int *)(v19 + 24) << 6;
  }
  if ( v34 == v38 + v37 )
    goto LABEL_30;
  sub_22C05A0((__int64)&v63, (unsigned __int8 *)(v34 + 24));
  v67 = 1;
LABEL_27:
  sub_22C6BD0(a2, a3, (unsigned __int8 *)&v63, a5, v26, v27);
  v28 = v67 == 0;
  *(_BYTE *)(a1 + 40) = 0;
  if ( !v28 )
  {
    sub_22C0650(a1, (unsigned __int8 *)&v63);
    v28 = v67 == 0;
    *(_BYTE *)(a1 + 40) = 1;
    if ( !v28 )
    {
      v67 = 0;
      sub_22C0090((unsigned __int8 *)&v63);
    }
  }
  return a1;
}
