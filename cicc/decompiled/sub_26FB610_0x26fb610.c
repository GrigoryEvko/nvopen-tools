// Function: sub_26FB610
// Address: 0x26fb610
//
__int64 __fastcall sub_26FB610(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned __int8 *a6)
{
  const __m128i *v7; // r14
  __int64 v8; // rcx
  __m128i v9; // xmm0
  bool v10; // zf
  _QWORD *v11; // rax
  __int64 result; // rax
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rax
  _QWORD *v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rax
  char v19; // al
  __int16 v20; // cx
  _QWORD *v21; // rax
  __int64 v22; // r15
  __int64 v23; // r9
  unsigned int *v24; // r12
  unsigned int *v25; // rbx
  __int64 v26; // rdx
  unsigned int v27; // esi
  _QWORD *v28; // r12
  __int64 v29; // rbx
  _QWORD *v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // rax
  char v33; // al
  __int16 v34; // cx
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // r15
  __int64 v37; // rbx
  unsigned int *v38; // rbx
  unsigned int *v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 (__fastcall *v42)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v43; // r9
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 v46; // r9
  __int64 v47; // r12
  __int64 v48; // r15
  _QWORD *v49; // rdi
  __int64 v50; // rbx
  unsigned int *v51; // r15
  unsigned int *v52; // rbx
  __int64 v53; // rdx
  unsigned int v54; // esi
  _BYTE *v55; // [rsp+8h] [rbp-1A8h]
  __int64 v60; // [rsp+30h] [rbp-180h]
  __int64 v61; // [rsp+38h] [rbp-178h]
  __int64 v62; // [rsp+38h] [rbp-178h]
  __int64 v63; // [rsp+38h] [rbp-178h]
  __int64 v64; // [rsp+38h] [rbp-178h]
  __int16 v65; // [rsp+5Ch] [rbp-154h]
  __int16 v66; // [rsp+5Eh] [rbp-152h]
  const __m128i *v67; // [rsp+68h] [rbp-148h]
  __m128i v68; // [rsp+70h] [rbp-140h] BYREF
  _DWORD *v69; // [rsp+80h] [rbp-130h]
  _BYTE v70[32]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v71; // [rsp+B0h] [rbp-100h]
  _BYTE v72[32]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v73; // [rsp+E0h] [rbp-D0h]
  unsigned int *v74; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+F8h] [rbp-B8h]
  _BYTE v76[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v77; // [rsp+120h] [rbp-90h]
  __int64 v78; // [rsp+128h] [rbp-88h]
  __int64 v79; // [rsp+130h] [rbp-80h]
  __int64 v80; // [rsp+138h] [rbp-78h]
  void **v81; // [rsp+140h] [rbp-70h]
  void **v82; // [rsp+148h] [rbp-68h]
  __int64 v83; // [rsp+150h] [rbp-60h]
  int v84; // [rsp+158h] [rbp-58h]
  __int16 v85; // [rsp+15Ch] [rbp-54h]
  char v86; // [rsp+15Eh] [rbp-52h]
  __int64 v87; // [rsp+160h] [rbp-50h]
  __int64 v88; // [rsp+168h] [rbp-48h]
  void *v89; // [rsp+170h] [rbp-40h] BYREF
  void *v90; // [rsp+178h] [rbp-38h] BYREF

  v7 = *(const __m128i **)a2;
  v67 = *(const __m128i **)(a2 + 8);
  v60 = a1 + 176;
  if ( v67 != *(const __m128i **)a2 )
  {
    v55 = a3;
    v8 = (__int64)v76;
    while ( 1 )
    {
      v9 = _mm_loadu_si128(v7);
      v10 = *(_BYTE *)(a1 + 204) == 0;
      v68 = v9;
      v69 = (_DWORD *)v7[1].m128i_i64[0];
      if ( v10 )
        goto LABEL_12;
      v11 = *(_QWORD **)(a1 + 184);
      v8 = *(unsigned int *)(a1 + 196);
      a3 = &v11[v8];
      if ( v11 != a3 )
      {
        while ( v9.m128i_i64[1] != *v11 )
        {
          if ( a3 == ++v11 )
            goto LABEL_29;
        }
        goto LABEL_8;
      }
LABEL_29:
      if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 192) )
      {
LABEL_12:
        sub_C8CC70(v60, v9.m128i_i64[1], (__int64)a3, v8, a5, (__int64)a6);
        if ( (_BYTE)a3 )
          goto LABEL_13;
LABEL_8:
        v7 = (const __m128i *)((char *)v7 + 24);
        if ( v67 == v7 )
          break;
      }
      else
      {
        *(_DWORD *)(a1 + 196) = v8 + 1;
        *a3 = v9.m128i_i64[1];
        ++*(_QWORD *)(a1 + 176);
LABEL_13:
        v13 = v68.m128i_i64[1];
        v14 = *(_QWORD *)(v68.m128i_i64[1] + 8);
        v15 = sub_BD5C60(v68.m128i_i64[1]);
        v86 = 7;
        v80 = v15;
        v81 = &v89;
        v82 = &v90;
        v74 = (unsigned int *)v76;
        v85 = 512;
        v89 = &unk_49DA100;
        LOWORD(v79) = 0;
        v75 = 0x200000000LL;
        v90 = &unk_49DA0B0;
        v83 = 0;
        v84 = 0;
        v87 = 0;
        v88 = 0;
        v77 = 0;
        v78 = 0;
        sub_D5F1F0((__int64)&v74, v13);
        v73 = 257;
        v16 = sub_F7CA10((__int64 *)&v74, v68.m128i_i64[0], a5, (__int64)v72, 0);
        v17 = *(_DWORD *)(v14 + 8);
        v71 = 257;
        v61 = (__int64)v16;
        if ( v17 >> 8 != 1 )
        {
          v18 = sub_AA4E30(v77);
          v19 = sub_AE5020(v18, v14);
          HIBYTE(v20) = HIBYTE(v66);
          v73 = 257;
          LOBYTE(v20) = v19;
          v66 = v20;
          v21 = sub_BD2C40(80, unk_3F10A14);
          v22 = (__int64)v21;
          if ( v21 )
            sub_B4D190((__int64)v21, v14, v61, (__int64)v72, 0, v66, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v82 + 2))(v82, v22, v70, v78, v79);
          v24 = v74;
          v25 = &v74[4 * (unsigned int)v75];
          if ( v74 != v25 )
          {
            do
            {
              v26 = *((_QWORD *)v24 + 1);
              v27 = *v24;
              v24 += 4;
              sub_B99FD0(v22, v27, v26);
            }
            while ( v25 != v24 );
          }
          if ( *(_BYTE *)(a1 + 104) )
            sub_26F96D0(
              (__int64)&v68,
              "virtual-const-prop",
              18,
              v55,
              a4,
              v23,
              *(__int64 (__fastcall **)(__int64, __int64))(a1 + 112),
              *(_QWORD *)(a1 + 120));
          sub_BD84D0(v68.m128i_i64[1], v22);
          v28 = (_QWORD *)v68.m128i_i64[1];
          if ( *(_BYTE *)v68.m128i_i64[1] == 34 )
          {
            v29 = *(_QWORD *)(v68.m128i_i64[1] - 96);
            v62 = v68.m128i_i64[1] + 24;
            v30 = sub_BD2C40(72, 1u);
            if ( v30 )
              sub_B4C8F0((__int64)v30, v29, 1u, v62, 0);
LABEL_23:
            sub_AA5980(*(v28 - 8), v28[5], 0);
            v28 = (_QWORD *)v68.m128i_i64[1];
            goto LABEL_24;
          }
          goto LABEL_24;
        }
        v31 = *(_QWORD *)(a1 + 56);
        v32 = sub_AA4E30(v77);
        v33 = sub_AE5020(v32, v31);
        HIBYTE(v34) = HIBYTE(v65);
        LOBYTE(v34) = v33;
        v65 = v34;
        v73 = 257;
        v35 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
        v36 = v35;
        if ( v35 )
          sub_B4D190((__int64)v35, v31, v61, (__int64)v72, 0, v65, 0, 0);
        (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v82 + 2))(
          v82,
          v36,
          v70,
          v78,
          v79);
        v37 = 4LL * (unsigned int)v75;
        if ( v74 != &v74[v37] )
        {
          v38 = &v74[v37];
          v39 = v74;
          do
          {
            v40 = *((_QWORD *)v39 + 1);
            v41 = *v39;
            v39 += 4;
            sub_B99FD0((__int64)v36, v41, v40);
          }
          while ( v38 != v39 );
        }
        v71 = 257;
        v42 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v81 + 2);
        if ( v42 != sub_9202E0 )
        {
          v43 = v42((__int64)v81, 28u, v36, a6);
          goto LABEL_41;
        }
        if ( *v36 <= 0x15u && *a6 <= 0x15u )
        {
          if ( (unsigned __int8)sub_AC47B0(28) )
            v43 = sub_AD5570(28, (__int64)v36, a6, 0, 0);
          else
            v43 = sub_AABE40(0x1Cu, v36, a6);
LABEL_41:
          if ( v43 )
            goto LABEL_42;
        }
        v73 = 257;
        v64 = sub_B504D0(28, (__int64)v36, (__int64)a6, (__int64)v72, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v82 + 2))(v82, v64, v70, v78, v79);
        v43 = v64;
        v50 = 4LL * (unsigned int)v75;
        v51 = &v74[v50];
        if ( v74 != &v74[v50] )
        {
          v52 = v74;
          do
          {
            v53 = *((_QWORD *)v52 + 1);
            v54 = *v52;
            v52 += 4;
            sub_B99FD0(v64, v54, v53);
          }
          while ( v51 != v52 );
          v43 = v64;
        }
LABEL_42:
        v44 = *(_QWORD *)(a1 + 56);
        v63 = v43;
        v73 = 257;
        v45 = (_BYTE *)sub_ACD640(v44, 0, 0);
        v47 = sub_92B530(&v74, 0x21u, v63, v45, (__int64)v72);
        if ( *(_BYTE *)(a1 + 104) )
          sub_26F96D0(
            (__int64)&v68,
            "virtual-const-prop-1-bit",
            24,
            v55,
            a4,
            v46,
            *(__int64 (__fastcall **)(__int64, __int64))(a1 + 112),
            *(_QWORD *)(a1 + 120));
        sub_BD84D0(v68.m128i_i64[1], v47);
        v28 = (_QWORD *)v68.m128i_i64[1];
        if ( *(_BYTE *)v68.m128i_i64[1] == 34 )
        {
          v48 = *(_QWORD *)(v68.m128i_i64[1] - 96);
          v49 = sub_BD2C40(72, 1u);
          if ( v49 )
            sub_B4C8F0((__int64)v49, v48, 1u, (__int64)(v28 + 3), 0);
          goto LABEL_23;
        }
LABEL_24:
        sub_B43D60(v28);
        if ( v69 )
          --*v69;
        nullsub_61();
        v89 = &unk_49DA100;
        nullsub_63();
        if ( v74 == (unsigned int *)v76 )
          goto LABEL_8;
        _libc_free((unsigned __int64)v74);
        v7 = (const __m128i *)((char *)v7 + 24);
        if ( v67 == v7 )
          break;
      }
    }
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
