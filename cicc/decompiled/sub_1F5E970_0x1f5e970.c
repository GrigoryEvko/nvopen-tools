// Function: sub_1F5E970
// Address: 0x1f5e970
//
__int64 __fastcall sub_1F5E970(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  __int64 *v16; // r12
  _QWORD *v17; // rax
  __int64 v18; // rcx
  _BYTE *v19; // rax
  __int64 v20; // rsi
  _BYTE *v21; // rdx
  __int64 v22; // rsi
  _BYTE *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r13
  const char *v31; // rsi
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 *v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // r13d
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rcx
  double v41; // xmm4_8
  double v42; // xmm5_8
  __int64 v43; // rdx
  unsigned __int64 v44; // r15
  __int64 v45; // r12
  __int64 v46; // r13
  __int64 i; // r14
  char v49; // al
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 *v54; // [rsp+18h] [rbp-1C8h]
  _BYTE *v55; // [rsp+18h] [rbp-1C8h]
  __int64 v56[2]; // [rsp+20h] [rbp-1C0h] BYREF
  _QWORD v57[2]; // [rsp+30h] [rbp-1B0h] BYREF
  __int64 v58[3]; // [rsp+40h] [rbp-1A0h] BYREF
  _QWORD *v59; // [rsp+58h] [rbp-188h]
  __int64 v60; // [rsp+60h] [rbp-180h]
  int v61; // [rsp+68h] [rbp-178h]
  __int64 v62; // [rsp+70h] [rbp-170h]
  __int64 v63; // [rsp+78h] [rbp-168h]
  __int64 *v64; // [rsp+90h] [rbp-150h] BYREF
  __int64 v65; // [rsp+98h] [rbp-148h]
  _BYTE v66[128]; // [rsp+A0h] [rbp-140h] BYREF
  _BYTE *v67; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+128h] [rbp-B8h]
  _BYTE v69[176]; // [rsp+130h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a2 + 80);
  v64 = (__int64 *)v66;
  v65 = 0x1000000000LL;
  v67 = v69;
  v68 = 0x1000000000LL;
  if ( v10 == a2 + 72 )
  {
    LODWORD(v35) = 0;
  }
  else
  {
    v11 = 0x40018000000001LL;
    do
    {
      v13 = 0;
      if ( v10 )
        v13 = v10 - 24;
      v14 = v13;
      v15 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v13) + 16) - 34;
      if ( (unsigned int)v15 <= 0x36 && _bittest64(&v11, v15) )
      {
        v49 = *(_BYTE *)(sub_157ED20(v14) + 16);
        if ( v49 == 74 )
        {
          v53 = (unsigned int)v65;
          if ( (unsigned int)v65 >= HIDWORD(v65) )
          {
            sub_16CD150((__int64)&v64, v66, 0, 8, v50, v51);
            v53 = (unsigned int)v65;
          }
          v64[v53] = v14;
          LODWORD(v65) = v65 + 1;
        }
        else if ( v49 == 73 )
        {
          v52 = (unsigned int)v68;
          if ( (unsigned int)v68 >= HIDWORD(v68) )
          {
            sub_16CD150((__int64)&v67, v69, 0, 8, v50, v51);
            v52 = (unsigned int)v68;
          }
          *(_QWORD *)&v67[8 * v52] = v14;
          LODWORD(v68) = v68 + 1;
        }
      }
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( a2 + 72 != v10 );
    if ( (_DWORD)v65 || (_DWORD)v68 )
    {
      v16 = *(__int64 **)(a2 + 40);
      v17 = (_QWORD *)sub_15E0530(a2);
      v18 = a1[20];
      v59 = v17;
      memset(v58, 0, sizeof(v58));
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v19 = (_BYTE *)sub_1632210((__int64)v16, (__int64)"__wasm_lpad_context", 19, v18);
      a1[21] = v19;
      v20 = a1[20];
      v56[0] = (__int64)"lpad_index_gep";
      LOWORD(v57[0]) = 259;
      a1[22] = sub_18174F0((__int64)v58, v20, v19, 0, 0, v56);
      v21 = (_BYTE *)a1[21];
      v22 = a1[20];
      v56[0] = (__int64)"lsda_gep";
      LOWORD(v57[0]) = 259;
      a1[23] = sub_18174F0((__int64)v58, v22, v21, 0, 1u, v56);
      v23 = (_BYTE *)a1[21];
      v24 = a1[20];
      v56[0] = (__int64)"selector_gep";
      LOWORD(v57[0]) = 259;
      a1[24] = sub_18174F0((__int64)v58, v24, v23, 0, 2u, v56);
      a1[25] = sub_15E26F0(v16, 6290, 0, 0);
      a1[26] = sub_15E26F0(v16, 6295, 0, 0);
      a1[27] = sub_15E26F0(v16, 6296, 0, 0);
      a1[28] = sub_15E26F0(v16, 6293, 0, 0);
      v25 = sub_15E26F0(v16, 6292, 0, 0);
      v26 = v59;
      a1[29] = v25;
      v27 = sub_16471D0(v26, 0);
      v28 = (__int64 *)sub_1643350(v59);
      v57[0] = v27;
      v56[0] = (__int64)v57;
      v56[1] = 0x100000001LL;
      v29 = sub_1644EA0(v28, v57, 1, 0);
      v30 = sub_1632080((__int64)v16, (__int64)"_Unwind_CallPersonality", 23, v29, 0);
      if ( (_QWORD *)v56[0] != v57 )
        _libc_free(v56[0]);
      a1[30] = v30;
      sub_15E0D50(v30, -1, 30);
      v31 = "__clang_call_terminate";
      v32 = sub_16321A0((__int64)v16, (__int64)"__clang_call_terminate", 22);
      v35 = v64;
      v36 = (unsigned int)v65;
      a1[31] = v32;
      v54 = &v35[v36];
      if ( v35 != v54 )
      {
        v37 = 0;
        do
        {
          while ( 1 )
          {
            v38 = *v35;
            v39 = sub_157ED20(*v35);
            v43 = *(_DWORD *)(v39 + 20) & 0xFFFFFFF;
            if ( (_DWORD)v43 == 2 && sub_1593BB0(*(_QWORD *)(v39 - 48), (__int64)v31, v43, v40) )
              break;
            v31 = (const char *)v38;
            sub_1F5DE70((__int64)a1, v38, v37++, a3, a4, a5, a6, v41, v42, a9, a10);
            if ( v54 == ++v35 )
              goto LABEL_18;
          }
          v31 = (const char *)v38;
          ++v35;
          sub_1F5DE70((__int64)a1, v38, 0xFFFFFFFF, a3, a4, a5, a6, v41, v42, a9, a10);
        }
        while ( v54 != v35 );
LABEL_18:
        v32 = a1[31];
      }
      if ( v32 )
      {
        v44 = (unsigned __int64)v67;
        v55 = &v67[8 * (unsigned int)v68];
        if ( v67 != v55 )
        {
          do
          {
            v45 = *(_QWORD *)v44;
            v46 = *(_QWORD *)(*(_QWORD *)v44 + 48LL);
            for ( i = *(_QWORD *)v44 + 40LL; i != v46; v46 = *(_QWORD *)(v46 + 8) )
            {
              if ( !v46 )
                BUG();
              if ( *(_BYTE *)(v46 - 8) == 78 && a1[31] == *(_QWORD *)(v46 - 48) )
                sub_1F5DE70((__int64)a1, v45, 0xFFFFFFFF, a3, a4, a5, a6, v33, v34, a9, a10);
            }
            v44 += 8LL;
          }
          while ( v55 != (_BYTE *)v44 );
        }
        LODWORD(v35) = 1;
      }
      else
      {
        LOBYTE(v35) = (_DWORD)v65 != 0;
      }
      if ( v58[0] )
        sub_161E7C0((__int64)v58, v58[0]);
    }
    else
    {
      LODWORD(v35) = 0;
    }
    if ( v67 != v69 )
      _libc_free((unsigned __int64)v67);
  }
  if ( v64 != (__int64 *)v66 )
    _libc_free((unsigned __int64)v64);
  return (unsigned int)v35;
}
