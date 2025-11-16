// Function: sub_28C3840
// Address: 0x28c3840
//
unsigned __int8 *__fastcall sub_28C3840(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 ***a5,
        __int64 a6)
{
  __int64 v8; // rbx
  int v9; // eax
  __int64 *i; // r13
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  char v24; // al
  __int64 *v25; // rax
  unsigned __int8 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r13
  unsigned int *v33; // rax
  __int32 v34; // ecx
  unsigned int *v35; // rdx
  char v36; // r13
  __int64 v37; // rax
  __int64 v38; // rdx
  char v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int64 v42; // r13
  __int64 **v43; // rax
  __int64 v44; // r9
  __int64 ***v45; // rax
  __int64 v46; // r13
  __int64 ***v47; // rax
  __int64 v48; // rsi
  char v49; // al
  _QWORD *v51; // rax
  __int64 v52; // rbx
  __int64 v53; // r14
  __int64 v54; // r13
  __int64 v55; // rdx
  unsigned int v56; // esi
  unsigned __int64 v57; // rsi
  __int64 v58; // [rsp+8h] [rbp-198h]
  __int64 **v59; // [rsp+8h] [rbp-198h]
  unsigned __int64 v60; // [rsp+18h] [rbp-188h]
  unsigned __int64 v61; // [rsp+20h] [rbp-180h]
  __int64 v64; // [rsp+30h] [rbp-170h]
  __int64 v65; // [rsp+40h] [rbp-160h]
  unsigned __int64 v66; // [rsp+40h] [rbp-160h]
  __int64 v67; // [rsp+40h] [rbp-160h]
  __int64 v68; // [rsp+40h] [rbp-160h]
  __int64 ***v69; // [rsp+48h] [rbp-158h] BYREF
  char v70[32]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v71; // [rsp+70h] [rbp-130h]
  unsigned __int64 v72; // [rsp+80h] [rbp-120h] BYREF
  __int64 v73; // [rsp+88h] [rbp-118h]
  __int16 v74; // [rsp+A0h] [rbp-100h]
  _BYTE *v75; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+B8h] [rbp-E8h]
  _BYTE v77[32]; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v78; // [rsp+E0h] [rbp-C0h] BYREF
  _QWORD v79[4]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+110h] [rbp-90h]
  __int64 v81; // [rsp+118h] [rbp-88h]
  __int64 v82; // [rsp+120h] [rbp-80h]
  __int64 v83; // [rsp+128h] [rbp-78h]
  void **v84; // [rsp+130h] [rbp-70h]
  void **v85; // [rsp+138h] [rbp-68h]
  __int64 v86; // [rsp+140h] [rbp-60h]
  int v87; // [rsp+148h] [rbp-58h]
  __int16 v88; // [rsp+14Ch] [rbp-54h]
  char v89; // [rsp+14Eh] [rbp-52h]
  __int64 v90; // [rsp+150h] [rbp-50h]
  __int64 v91; // [rsp+158h] [rbp-48h]
  void *v92; // [rsp+160h] [rbp-40h] BYREF
  void *v93; // [rsp+168h] [rbp-38h] BYREF

  v8 = a2;
  v75 = v77;
  v76 = 0x400000000LL;
  v9 = *(_DWORD *)(a2 + 4);
  v69 = a5;
  for ( i = (__int64 *)(a2 + 32 * (1LL - (v9 & 0x7FFFFFF))); (__int64 *)a2 != i; LODWORD(v76) = v76 + 1 )
  {
    v13 = sub_DD8400(a1[3], *i);
    v14 = (unsigned int)v76;
    v15 = (unsigned int)v76 + 1LL;
    if ( v15 > HIDWORD(v76) )
    {
      sub_C8D5F0((__int64)&v75, v77, v15, 8u, v11, v12);
      v14 = (unsigned int)v76;
    }
    i += 4;
    *(_QWORD *)&v75[8 * v14] = v13;
  }
  v16 = sub_DD8400(a1[3], a4);
  *(_QWORD *)&v75[8 * a3] = v16;
  v17 = sub_D97090(
          a1[3],
          *(_QWORD *)(*(_QWORD *)(a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL));
  v18 = sub_D97090(a1[3], *(_QWORD *)(a4 + 8));
  v61 = sub_9208B0(a1[1], v18);
  v78.m128i_i64[1] = v19;
  v20 = a1[1];
  v78.m128i_i64[0] = v61;
  v21 = sub_9208B0(v20, v17);
  v22 = a1[2];
  v60 = v21;
  v23 = *a1;
  v78 = (__m128i)(unsigned __int64)a1[1];
  v79[1] = v22;
  v79[2] = v23;
  v79[0] = 0;
  v79[3] = a2;
  v80 = 0;
  v81 = 0;
  LOWORD(v82) = 257;
  v24 = sub_9AC470(a4, &v78, 0);
  if ( v61 < v60 && v24 )
  {
    v51 = sub_DC2B70(a1[3], *(_QWORD *)&v75[8 * a3], v17, 0);
    *(_QWORD *)&v75[8 * a3] = v51;
  }
  v25 = sub_DD8EB0((__int64 *)a1[3], a2, (__int64)&v75);
  v26 = sub_28C1F40((__int64)a1, (__int64)v25, a2);
  if ( v26 )
  {
    v27 = sub_BD5C60(a2);
    v89 = 7;
    v83 = v27;
    v84 = &v92;
    v85 = &v93;
    v78.m128i_i64[0] = (__int64)v79;
    v92 = &unk_49DA100;
    v78.m128i_i64[1] = 0x200000000LL;
    v86 = 0;
    v93 = &unk_49DA0B0;
    v28 = *(_QWORD *)(a2 + 40);
    v87 = 0;
    v80 = v28;
    v88 = 512;
    v90 = 0;
    v91 = 0;
    v81 = a2 + 24;
    LOWORD(v82) = 0;
    v29 = *(_QWORD *)sub_B46C60(a2);
    v72 = v29;
    if ( v29 && (sub_B96E90((__int64)&v72, v29, 1), (v32 = v72) != 0) )
    {
      v33 = (unsigned int *)v78.m128i_i64[0];
      v34 = v78.m128i_i32[2];
      v35 = (unsigned int *)(v78.m128i_i64[0] + 16LL * v78.m128i_u32[2]);
      if ( (unsigned int *)v78.m128i_i64[0] != v35 )
      {
        while ( 1 )
        {
          v31 = *v33;
          if ( !(_DWORD)v31 )
            break;
          v33 += 4;
          if ( v35 == v33 )
            goto LABEL_32;
        }
        *((_QWORD *)v33 + 1) = v72;
LABEL_16:
        sub_B91220((__int64)&v72, v32);
LABEL_19:
        v65 = a1[1];
        v36 = sub_AE5020(v65, a6);
        v37 = sub_9208B0(v65, a6);
        v73 = v38;
        v72 = ((1LL << v36) + ((unsigned __int64)(v37 + 7) >> 3) - 1) >> v36 << v36;
        v66 = sub_CA1930(&v72);
        v58 = *(_QWORD *)(v8 + 80);
        v64 = a1[1];
        v39 = sub_AE5020(v64, v58);
        v40 = sub_9208B0(v64, v58);
        v73 = v41;
        v72 = ((1LL << v39) + ((unsigned __int64)(v40 + 7) >> 3) - 1) >> v39 << v39;
        v42 = sub_CA1930(&v72);
        if ( v66 % v42 )
        {
          v26 = 0;
        }
        else
        {
          v43 = (__int64 **)sub_AE4570(a1[1], *(_QWORD *)(v8 + 8));
          v44 = (__int64)v43;
          if ( v43 != v69[1] )
          {
            v74 = 257;
            v59 = v43;
            v45 = (__int64 ***)sub_2784C30(v78.m128i_i64, (unsigned __int64)v69, v43, (__int64)&v72);
            v44 = (__int64)v59;
            v69 = v45;
          }
          if ( v66 != v42 )
          {
            v71 = 257;
            v46 = sub_AD64C0(v44, v66 / v42, 0);
            v47 = (__int64 ***)(*((__int64 (__fastcall **)(void **, __int64, __int64 ***, __int64, _QWORD, _QWORD))*v84
                                + 4))(
                                 v84,
                                 17,
                                 v69,
                                 v46,
                                 0,
                                 0);
            if ( !v47 )
            {
              v74 = 257;
              v67 = sub_B504D0(17, (__int64)v69, v46, (__int64)&v72, 0, 0);
              (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v85 + 2))(v85, v67, v70, v81, v82);
              v47 = (__int64 ***)v67;
              if ( v78.m128i_i64[0] != v78.m128i_i64[0] + 16LL * v78.m128i_u32[2] )
              {
                v68 = v8;
                v52 = v78.m128i_i64[0];
                v53 = v78.m128i_i64[0] + 16LL * v78.m128i_u32[2];
                v54 = (__int64)v47;
                do
                {
                  v55 = *(_QWORD *)(v52 + 8);
                  v56 = *(_DWORD *)v52;
                  v52 += 16;
                  sub_B99FD0(v54, v56, v55);
                }
                while ( v53 != v52 );
                v8 = v68;
                v47 = (__int64 ***)v54;
              }
            }
            v69 = v47;
          }
          v48 = *(_QWORD *)(v8 + 80);
          v74 = 257;
          v26 = (unsigned __int8 *)sub_921130(
                                     (unsigned int **)&v78,
                                     v48,
                                     (__int64)v26,
                                     (_BYTE **)&v69,
                                     1,
                                     (__int64)&v72,
                                     0);
          v49 = sub_B4DE30(v8);
          sub_B4DE00((__int64)v26, v49);
          sub_BD6B90(v26, (unsigned __int8 *)v8);
        }
        nullsub_61();
        v92 = &unk_49DA100;
        nullsub_63();
        if ( (_QWORD *)v78.m128i_i64[0] != v79 )
          _libc_free(v78.m128i_u64[0]);
        goto LABEL_28;
      }
LABEL_32:
      if ( v78.m128i_u32[2] >= (unsigned __int64)v78.m128i_u32[3] )
      {
        v57 = v78.m128i_u32[2] + 1LL;
        if ( v78.m128i_u32[3] < v57 )
        {
          sub_C8D5F0((__int64)&v78, v79, v57, 0x10u, v30, v31);
          v35 = (unsigned int *)(v78.m128i_i64[0] + 16LL * v78.m128i_u32[2]);
        }
        *(_QWORD *)v35 = 0;
        *((_QWORD *)v35 + 1) = v32;
        v32 = v72;
        ++v78.m128i_i32[2];
      }
      else
      {
        if ( v35 )
        {
          *v35 = 0;
          *((_QWORD *)v35 + 1) = v32;
          v34 = v78.m128i_i32[2];
          v32 = v72;
        }
        v78.m128i_i32[2] = v34 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v78, 0);
      v32 = v72;
    }
    if ( !v32 )
      goto LABEL_19;
    goto LABEL_16;
  }
LABEL_28:
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
  return v26;
}
