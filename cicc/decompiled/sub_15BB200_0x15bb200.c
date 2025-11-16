// Function: sub_15BB200
// Address: 0x15bb200
//
__int64 __fastcall sub_15BB200(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 v8; // r15
  char v9; // r11
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // r9d
  int v21; // r9d
  unsigned int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // rcx
  __int8 *v25; // rax
  __int8 *v26; // rax
  unsigned __int64 v27; // rsi
  int v28; // ecx
  __int64 v29; // rcx
  unsigned __int64 v30; // rcx
  unsigned int v31; // esi
  __int64 v32; // rcx
  __int64 *v33; // rsi
  unsigned int v34; // ecx
  __int64 v35; // rcx
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rdx
  char v38; // [rsp+Ch] [rbp-104h]
  __int64 v39; // [rsp+10h] [rbp-100h]
  __int64 v41; // [rsp+18h] [rbp-F8h]
  int v42; // [rsp+18h] [rbp-F8h]
  int v43; // [rsp+20h] [rbp-F0h]
  __int64 *v44; // [rsp+20h] [rbp-F0h]
  __int8 *v45; // [rsp+20h] [rbp-F0h]
  _QWORD *src; // [rsp+28h] [rbp-E8h]
  int v48; // [rsp+30h] [rbp-E0h]
  int i; // [rsp+30h] [rbp-E0h]
  __int64 v50; // [rsp+38h] [rbp-D8h]
  __int64 v51; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v53; // [rsp+50h] [rbp-C0h]
  __int64 v54; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int64 v56; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v57; // [rsp+A8h] [rbp-68h]
  __int64 v58; // [rsp+B0h] [rbp-60h]
  __int64 v59; // [rsp+B8h] [rbp-58h]
  __int64 v60; // [rsp+C0h] [rbp-50h]
  __int64 v61; // [rsp+C8h] [rbp-48h]
  __int64 v62; // [rsp+D0h] [rbp-40h]
  __int64 v63; // [rsp+D8h] [rbp-38h]

  if ( a4 )
    goto LABEL_4;
  v8 = *a1;
  v53 = (_QWORD *)a2;
  v9 = a5;
  v54 = a3;
  v50 = *(_QWORD *)(v8 + 664);
  if ( !*(_DWORD *)(v8 + 680) )
    goto LABEL_3;
  if ( *(_BYTE *)a2 == 1 )
  {
    v15 = *(_QWORD *)(a2 + 136);
    v16 = *(_DWORD *)(v15 + 32);
    v17 = *(__int64 **)(v15 + 24);
    v18 = v16 > 0x40 ? *v17 : (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
    v48 = *(_DWORD *)(v8 + 680);
    dest[0].m128i_i64[0] = v18;
    LODWORD(v19) = sub_15B15B0(dest, &v54);
    v20 = v48;
    v9 = a5;
  }
  else
  {
    v43 = *(_DWORD *)(v8 + 680);
    v63 = sub_15AF870();
    v51 = 0;
    v25 = sub_15B2320(dest, &v51, dest[0].m128i_i8, (unsigned __int64)&v56, (__int64)v53);
    v52 = v51;
    v26 = sub_15B2220(dest, &v52, v25, (unsigned __int64)&v56, v54);
    if ( v52 )
    {
      v38 = a5;
      v39 = v52;
      v42 = v43;
      v45 = v26;
      sub_15AF6E0(dest[0].m128i_i8, v26, (char *)&v56);
      sub_1593A20(&v56, dest);
      v9 = v38;
      v20 = v42;
      v36 = 0xB492B66FBE98F273LL
          * (((unsigned __int64)(v45 - (__int8 *)dest + v39) >> 47) ^ (v45 - (__int8 *)dest + v39))
          + v56
          - 0x622015F714C7D297LL
          * (((0x9DDFEA08EB382D69LL
             * (((0x9DDFEA08EB382D69LL * (v62 ^ v60)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v62 ^ v60)) ^ v62)) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL * (v62 ^ v60)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v62 ^ v60)) ^ v62)));
      v37 = 0x9DDFEA08EB382D69LL
          * (v36
           ^ (0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL
               * ((0x9DDFEA08EB382D69LL * (v61 ^ v59)) ^ v61 ^ ((0x9DDFEA08EB382D69LL * (v61 ^ v59)) >> 47))) >> 47)
             ^ (0x9DDFEA08EB382D69LL
              * ((0x9DDFEA08EB382D69LL * (v61 ^ v59)) ^ v61 ^ ((0x9DDFEA08EB382D69LL * (v61 ^ v59)) >> 47))))
            + v58
            - 0x4B6D499041670D8DLL * (v57 ^ (v57 >> 47))));
      v19 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v36 ^ v37 ^ (v37 >> 47)))
           ^ ((0x9DDFEA08EB382D69LL * (v36 ^ v37 ^ (v37 >> 47))) >> 47));
    }
    else
    {
      LODWORD(v19) = sub_1593600(dest, v26 - (__int8 *)dest, v63);
      v20 = v43;
      v9 = a5;
    }
  }
  v21 = v20 - 1;
  v22 = v21 & v19;
  v23 = (__int64 *)(v50 + 8LL * v22);
  v24 = *v23;
  if ( *v23 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v24 == -16 || v54 != *(_QWORD *)(v24 + 24) )
      goto LABEL_16;
    v27 = *(_QWORD *)(v24 - 8LL * *(unsigned int *)(v24 + 8));
    v28 = *(unsigned __int8 *)v27;
    src = (_QWORD *)v27;
    if ( (_BYTE)v28 == 1 )
    {
      v29 = *(_QWORD *)(v27 + 136);
    }
    else
    {
      if ( (unsigned int)(v28 - 24) > 1 )
        goto LABEL_31;
      v29 = v27 | 4;
    }
    if ( (v29 & 4) == 0 )
    {
      v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v30 )
      {
        if ( *(_BYTE *)v53 == 1 )
        {
          v31 = *(_DWORD *)(v30 + 32);
          v44 = *(__int64 **)(v30 + 24);
          v41 = v31 > 0x40 ? *v44 : (__int64)((_QWORD)v44 << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31);
          v32 = v53[17];
          v33 = *(__int64 **)(v32 + 24);
          v34 = *(_DWORD *)(v32 + 32);
          v35 = v34 > 0x40 ? *v33 : (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v34)) >> (64 - (unsigned __int8)v34);
          if ( v35 == v41 )
            break;
        }
      }
    }
LABEL_31:
    if ( v53 == src )
      break;
LABEL_16:
    v22 = v21 & (i + v22);
    v23 = (__int64 *)(v50 + 8LL * v22);
    v24 = *v23;
    if ( *v23 == -8 )
      goto LABEL_3;
  }
  if ( v23 != (__int64 *)(*(_QWORD *)(v8 + 664) + 8LL * *(unsigned int *)(v8 + 680)) )
  {
    result = *v23;
    if ( *v23 )
      return result;
  }
LABEL_3:
  result = 0;
  if ( !v9 )
    return result;
LABEL_4:
  v11 = *a1;
  dest[0].m128i_i64[0] = a2;
  v12 = v11 + 656;
  v13 = sub_161E980(32, 1);
  v14 = v13;
  if ( v13 )
  {
    sub_1623D80(v13, (_DWORD)a1, 9, a4, (unsigned int)dest, 1, 0, 0);
    *(_QWORD *)(v14 + 24) = a3;
    *(_WORD *)(v14 + 2) = 33;
  }
  return sub_15BB120(v14, a4, v12);
}
