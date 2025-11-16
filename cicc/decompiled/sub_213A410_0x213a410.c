// Function: sub_213A410
// Address: 0x213a410
//
__int64 *__fastcall sub_213A410(__int64 *a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v3; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 v13; // r10
  _QWORD *v14; // rax
  __int64 v15; // r11
  __int128 *v16; // rcx
  __int64 v17; // r8
  unsigned __int8 v18; // bl
  unsigned __int16 v19; // si
  __int64 *v20; // rbx
  const __m128i *v21; // r9
  unsigned int i; // r12d
  __int64 v23; // rdx
  __int64 v25; // r13
  unsigned __int8 *v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // ebx
  __int64 v29; // r15
  __int64 v30; // rax
  unsigned int v31; // eax
  unsigned int v32; // ebx
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // r10
  _QWORD *v40; // r13
  __int64 v41; // r9
  __int64 v42; // r11
  __int128 *v43; // rbx
  __int64 v44; // r8
  unsigned __int8 v45; // cl
  __int64 *v46; // rbx
  const __m128i *v47; // r9
  const __m128i *v48; // r9
  __int64 v49; // [rsp+0h] [rbp-A0h]
  __int64 v50; // [rsp+8h] [rbp-98h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  __int128 *v52; // [rsp+18h] [rbp-88h]
  __int128 v53; // [rsp+20h] [rbp-80h]
  __int64 v54; // [rsp+20h] [rbp-80h]
  __int64 v55; // [rsp+28h] [rbp-78h]
  __int64 v56; // [rsp+30h] [rbp-70h]
  unsigned __int8 v57; // [rsp+30h] [rbp-70h]
  _QWORD *v58; // [rsp+38h] [rbp-68h]
  __int64 v59; // [rsp+38h] [rbp-68h]
  __int64 v60; // [rsp+38h] [rbp-68h]
  __int128 v61; // [rsp+40h] [rbp-60h]
  unsigned int v62; // [rsp+40h] [rbp-60h]
  __int64 (__fastcall *v63)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+40h] [rbp-60h]
  unsigned __int8 v64; // [rsp+40h] [rbp-60h]
  __int64 v65; // [rsp+40h] [rbp-60h]
  __int64 v66; // [rsp+50h] [rbp-50h] BYREF
  int v67; // [rsp+58h] [rbp-48h]
  __int64 v68; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 32);
  if ( a3 == 1 )
  {
    v25 = *a1;
    v26 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 40LL) + 16LL * *(unsigned int *)(v5 + 88));
    v27 = a1[1];
    v28 = *v26;
    v29 = *(_QWORD *)(v27 + 48);
    v59 = *((_QWORD *)v26 + 1);
    v63 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 264LL);
    v30 = sub_1E0A0C0(*(_QWORD *)(v27 + 32));
    v31 = v63(v25, v30, v29, v28, v59);
    v64 = v31;
    v32 = v31;
    v34 = v33;
    sub_1F40D10(
      (__int64)&v66,
      *a1,
      *(_QWORD *)(a1[1] + 48),
      *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL));
    if ( !(_BYTE)v32 || (v35 = v64, !*(_QWORD *)(*a1 + 8LL * (unsigned __int8)v32 + 120)) )
    {
      v34 = v68;
      v35 = (unsigned __int8)v67;
    }
    LOBYTE(v32) = v35;
    v36 = sub_1D25E70(
            a1[1],
            **(unsigned __int8 **)(a2 + 40),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
            v32,
            v34,
            v35,
            1,
            0);
    v38 = *(_QWORD *)(a2 + 72);
    v39 = v36;
    v40 = (_QWORD *)a1[1];
    v41 = *(_QWORD *)(a2 + 104);
    v42 = v37;
    v43 = *(__int128 **)(a2 + 32);
    v44 = *(_QWORD *)(a2 + 96);
    v66 = v38;
    v45 = *(_BYTE *)(a2 + 88);
    if ( v38 )
    {
      v55 = v37;
      v57 = *(_BYTE *)(a2 + 88);
      v54 = v36;
      v60 = v41;
      v65 = v44;
      sub_1623A60((__int64)&v66, v38, 2);
      v45 = v57;
      v39 = v54;
      v42 = v55;
      v41 = v60;
      v44 = v65;
    }
    v67 = *(_DWORD *)(a2 + 64);
    v46 = sub_1D24690(
            v40,
            0xDEu,
            (__int64)&v66,
            v45,
            v44,
            v41,
            v39,
            v42,
            *v43,
            *(__int128 *)((char *)v43 + 40),
            v43[5],
            *(__int128 *)((char *)v43 + 120));
    if ( v66 )
      sub_161E7C0((__int64)&v66, v66);
    sub_2013400((__int64)a1, a2, 0, (__int64)v46, 0, v47);
    sub_2013400((__int64)a1, a2, 2, (__int64)v46, (__m128i *)2, v48);
    return v46;
  }
  else
  {
    *(_QWORD *)&v61 = sub_2138AD0((__int64)a1, *(_QWORD *)(v5 + 80), *(_QWORD *)(v5 + 88));
    *((_QWORD *)&v61 + 1) = v6;
    *(_QWORD *)&v53 = sub_2138AD0(
                        (__int64)a1,
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL));
    *((_QWORD *)&v53 + 1) = v7;
    v9 = sub_1D25E70(
           a1[1],
           *(unsigned __int8 *)(*(_QWORD *)(v61 + 40) + 16LL * DWORD2(v61)),
           *(_QWORD *)(*(_QWORD *)(v61 + 40) + 16LL * DWORD2(v61) + 8),
           *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 24LL),
           v8,
           1,
           0);
    v11 = *(_QWORD *)(a2 + 72);
    v12 = *(_QWORD *)(a2 + 104);
    v13 = v9;
    v14 = (_QWORD *)a1[1];
    v15 = v10;
    v16 = *(__int128 **)(a2 + 32);
    v17 = *(_QWORD *)(a2 + 96);
    v18 = *(_BYTE *)(a2 + 88);
    v66 = v11;
    v58 = v14;
    if ( v11 )
    {
      v49 = v13;
      v50 = v10;
      v51 = v12;
      v52 = v16;
      v56 = v17;
      sub_1623A60((__int64)&v66, v11, 2);
      v13 = v49;
      v17 = v56;
      v15 = v50;
      v12 = v51;
      v16 = v52;
    }
    v19 = *(_WORD *)(a2 + 24);
    v67 = *(_DWORD *)(a2 + 64);
    v20 = sub_1D24690(v58, v19, (__int64)&v66, v18, v17, v12, v13, v15, *v16, *(__int128 *)((char *)v16 + 40), v61, v53);
    if ( v66 )
      sub_161E7C0((__int64)&v66, v66);
    v62 = *(_DWORD *)(a2 + 60);
    if ( v62 > 1 )
    {
      for ( i = 1; i != v62; ++i )
      {
        v23 = i;
        v3 = v23 | v3 & 0xFFFFFFFF00000000LL;
        sub_2013400((__int64)a1, a2, v23, (__int64)v20, (__m128i *)v3, v21);
      }
    }
    return v20;
  }
}
