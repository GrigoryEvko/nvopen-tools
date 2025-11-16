// Function: sub_344B300
// Address: 0x344b300
//
__int64 __fastcall sub_344B300(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  int v11; // edx
  _QWORD *v12; // r13
  __int64 v14; // r8
  __int16 v15; // si
  __int64 v16; // r8
  __int64 *v17; // rsi
  __int128 v18; // xmm0
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned __int8 *v24; // r14
  __int64 v25; // rdx
  __int64 v26; // r15
  __int128 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdx
  int v30; // ecx
  unsigned __int8 *v31; // r14
  __int64 v32; // rdx
  __int64 v33; // r15
  __int128 v34; // rax
  __int128 v35; // rax
  __int64 v36; // r9
  __int128 v37; // rax
  __int64 v38; // r10
  unsigned int v39; // r11d
  __int128 v40; // rax
  __int64 v41; // r9
  __int128 v42; // [rsp-30h] [rbp-A0h]
  __int128 v43; // [rsp-30h] [rbp-A0h]
  __int128 v44; // [rsp-30h] [rbp-A0h]
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int128 v49; // [rsp+10h] [rbp-60h]
  __int128 v50; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  unsigned int v57; // [rsp+30h] [rbp-40h] BYREF
  __int64 v58; // [rsp+38h] [rbp-38h]

  v11 = *(_DWORD *)(a4 + 24);
  v12 = *(_QWORD **)(a10 + 16);
  v14 = *(_QWORD *)(a4 + 48) + 16LL * a5;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  LOWORD(v57) = v15;
  v17 = *(__int64 **)(a4 + 40);
  v58 = v16;
  v18 = (__int128)_mm_loadu_si128((const __m128i *)(v17 + 5));
  v19 = *v17;
  v20 = *((unsigned int *)v17 + 2);
  v21 = v17[5];
  v22 = *((unsigned int *)v17 + 12);
  if ( v19 == a7 && (_DWORD)v20 == (_DWORD)a8 )
  {
    v52 = a3;
    v24 = sub_3400BD0((__int64)v12, 0, a9, v57, v16, 0, (__m128i)v18, 0);
    v26 = v25;
    *(_QWORD *)&v27 = sub_33ED040(v12, a6);
    *((_QWORD *)&v42 + 1) = v26;
    *(_QWORD *)&v42 = v24;
    return sub_340F900(v12, 0xD0u, a9, a2, v52, v28, v18, v42, v27);
  }
  else
  {
    if ( v21 != a7 || (_DWORD)v22 != (_DWORD)a8 )
      return 0;
    if ( v11 == 56 || v11 == 188 )
    {
      v53 = a3;
      v31 = sub_3400BD0((__int64)v12, 0, a9, v57, v58, 0, (__m128i)v18, 0);
      v33 = v32;
      *(_QWORD *)&v49 = v19;
      *((_QWORD *)&v49 + 1) = v20;
      *(_QWORD *)&v34 = sub_33ED040(v12, a6);
      *((_QWORD *)&v43 + 1) = v33;
      *(_QWORD *)&v43 = v31;
      return sub_340F900(v12, 0xD0u, a9, a2, v53, v20, v49, v43, v34);
    }
    else
    {
      v29 = *(_QWORD *)(a4 + 56);
      if ( !v29 )
        return 0;
      v30 = 1;
      do
      {
        if ( a5 == *(_DWORD *)(v29 + 8) )
        {
          if ( !v30 )
            return 0;
          v29 = *(_QWORD *)(v29 + 32);
          if ( !v29 )
            goto LABEL_19;
          if ( a5 == *(_DWORD *)(v29 + 8) )
            return 0;
          v30 = 0;
        }
        v29 = *(_QWORD *)(v29 + 32);
      }
      while ( v29 );
      if ( v30 == 1 )
        return 0;
LABEL_19:
      v54 = a3;
      if ( sub_32844A0((unsigned __int16 *)&v57, v22) == 1 )
        return 0;
      *(_QWORD *)&v35 = sub_3400E40((__int64)v12, 1, v57, v58, a9, (__m128i)v18);
      *(_QWORD *)&v37 = sub_3406EB0(
                          v12,
                          0xBEu,
                          a9,
                          *(unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8),
                          *(_QWORD *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8 + 8),
                          v36,
                          v18,
                          v35);
      v38 = v54;
      v39 = a6;
      if ( !*(_BYTE *)(a10 + 12) )
      {
        v45 = *((_QWORD *)&v37 + 1);
        v51 = v37;
        sub_32C2500((__int64 *)a10, v37);
        *((_QWORD *)&v37 + 1) = v45;
        v39 = a6;
        v38 = v54;
        *(_QWORD *)&v37 = v51;
      }
      v55 = v38;
      v50 = v37;
      *(_QWORD *)&v40 = sub_33ED040(v12, v39);
      *((_QWORD *)&v44 + 1) = v20;
      *(_QWORD *)&v44 = v19;
      return sub_340F900(v12, 0xD0u, a9, a2, v55, v41, v44, v50, v40);
    }
  }
}
