// Function: sub_15B7360
// Address: 0x15b7360
//
__int64 __fastcall sub_15B7360(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int8 *v15; // rax
  __int8 *v16; // rax
  int v17; // r14d
  __int64 v18; // rsi
  int v19; // r8d
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // [rsp+8h] [rbp-E8h]
  __int8 *v27; // [rsp+18h] [rbp-D8h]
  __int64 v28; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v30; // [rsp+30h] [rbp-C0h]
  __int64 v31; // [rsp+38h] [rbp-B8h] BYREF
  __m128i src[4]; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v33; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v34; // [rsp+88h] [rbp-68h]
  __int64 v35; // [rsp+90h] [rbp-60h]
  __int64 v36; // [rsp+98h] [rbp-58h]
  __int64 v37; // [rsp+A0h] [rbp-50h]
  __int64 v38; // [rsp+A8h] [rbp-48h]
  __int64 v39; // [rsp+B0h] [rbp-40h]
  __int64 v40; // [rsp+B8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v8 = *(_QWORD *)(*a2 - 8LL * *(unsigned int *)(*a2 + 8));
    v9 = *(_QWORD *)(*a2 + 24);
    v30 = v8;
    v31 = v9;
    if ( *(_BYTE *)v8 == 1 )
    {
      v10 = *(_QWORD *)(v8 + 136);
      v11 = *(_DWORD *)(v10 + 32);
      v12 = *(__int64 **)(v10 + 24);
      if ( v11 <= 0x40 )
        v13 = (__int64)((_QWORD)v12 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
      else
        v13 = *v12;
      src[0].m128i_i64[0] = v13;
      LODWORD(v14) = sub_15B15B0(src, &v31);
    }
    else
    {
      v40 = sub_15AF870();
      v28 = 0;
      v15 = sub_15B2320(src, &v28, src[0].m128i_i8, (unsigned __int64)&v33, v30);
      v29 = v28;
      v16 = sub_15B2220(src, &v29, v15, (unsigned __int64)&v33, v31);
      if ( v29 )
      {
        v26 = v29;
        v27 = v16;
        sub_15AF6E0(src[0].m128i_i8, v16, (char *)&v33);
        sub_1593A20(&v33, src);
        v24 = 0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL
               * ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) ^ v39 ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) >> 47))) >> 47)
             ^ (0x9DDFEA08EB382D69LL
              * ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) ^ v39 ^ ((0x9DDFEA08EB382D69LL * (v39 ^ v37)) >> 47))))
            + v33
            - 0x4B6D499041670D8DLL
            * (((unsigned __int64)(v27 - (__int8 *)src + v26) >> 47) ^ (v27 - (__int8 *)src + v26));
        v25 = 0x9DDFEA08EB382D69LL
            * (v24
             ^ (0xB492B66FBE98F273LL * (v34 ^ (v34 >> 47))
              + v35
              - 0x622015F714C7D297LL
              * (((0x9DDFEA08EB382D69LL
                 * (((0x9DDFEA08EB382D69LL * (v38 ^ v36)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v38 ^ v36)) ^ v38)) >> 47)
               ^ (0x9DDFEA08EB382D69LL
                * (((0x9DDFEA08EB382D69LL * (v38 ^ v36)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v38 ^ v36)) ^ v38)))));
        v14 = 0x9DDFEA08EB382D69LL
            * ((0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24))
             ^ ((0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25 ^ v24)) >> 47));
      }
      else
      {
        LODWORD(v14) = sub_1593600(src, v16 - (__int8 *)src, v40);
      }
    }
    v17 = v4 - 1;
    v18 = *a2;
    v19 = 1;
    v20 = 0;
    v21 = v17 & v14;
    v22 = (_QWORD *)(v6 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == *a2 )
    {
LABEL_18:
      *a3 = v22;
      return 1;
    }
    else
    {
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v20 )
          v22 = v20;
        v21 = v17 & (v19 + v21);
        v23 = *(_QWORD *)(v6 + 8LL * v21);
        if ( v23 == v18 )
        {
          v22 = (_QWORD *)(v6 + 8LL * v21);
          goto LABEL_18;
        }
        ++v19;
        v20 = v22;
        v22 = (_QWORD *)(v6 + 8LL * v21);
      }
      if ( !v20 )
        v20 = v22;
      *a3 = v20;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
