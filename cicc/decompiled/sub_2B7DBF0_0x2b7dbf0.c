// Function: sub_2B7DBF0
// Address: 0x2b7dbf0
//
__int64 __fastcall sub_2B7DBF0(__int64 *a1, _QWORD *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r10
  _BYTE *v12; // r14
  unsigned __int64 v13; // rax
  char v14; // al
  __int64 v15; // r9
  _BYTE *v16; // rax
  __int64 v17; // r12
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // r13
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  int v31; // eax
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // r9
  int v34; // ecx
  unsigned __int64 v35; // rax
  __m128i *v36; // r8
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r9
  _QWORD *v42; // rax
  __int64 v43; // r10
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 v47; // rdx
  unsigned int v48; // esi
  unsigned __int64 v49; // rdx
  __int64 v50; // rdx
  __m128i *v51; // rax
  __int64 v52; // rdi
  const void *v53; // rsi
  __int64 v55; // [rsp+10h] [rbp-E0h]
  __int64 v56; // [rsp+20h] [rbp-D0h]
  __int64 v57; // [rsp+20h] [rbp-D0h]
  __int64 v58; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v59; // [rsp+20h] [rbp-D0h]
  __int64 v60; // [rsp+28h] [rbp-C8h]
  __int64 v61; // [rsp+28h] [rbp-C8h]
  __int64 v62; // [rsp+28h] [rbp-C8h]
  __int64 v63; // [rsp+28h] [rbp-C8h]
  __int64 v64; // [rsp+28h] [rbp-C8h]
  __int64 v65; // [rsp+28h] [rbp-C8h]
  __int64 v66; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v67[4]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v68; // [rsp+60h] [rbp-90h]
  __m128i v69; // [rsp+70h] [rbp-80h] BYREF
  __int64 v70; // [rsp+80h] [rbp-70h]
  __int64 v71; // [rsp+88h] [rbp-68h]
  __int64 v72; // [rsp+90h] [rbp-60h]
  __int64 v73; // [rsp+98h] [rbp-58h]
  __int64 v74; // [rsp+A0h] [rbp-50h]
  __int64 v75; // [rsp+A8h] [rbp-48h]
  __int16 v76; // [rsp+B0h] [rbp-40h]

  v5 = a3;
  v9 = *(_QWORD *)(a3 + 8);
  if ( a5 != v9 )
  {
    v10 = a5;
    v11 = *a1;
    if ( (unsigned __int8)(*(_BYTE *)a3 - 68) > 1u )
      goto LABEL_3;
    v12 = *(_BYTE **)(a3 - 32);
    if ( *v12 > 0x1Cu )
    {
      v57 = *a1;
      v69.m128i_i64[0] = *(_QWORD *)(a3 - 32);
      v19 = sub_2B4B3F0(v11 + 1976, v69.m128i_i64);
      v11 = v57;
      v10 = a5;
      if ( v19 || (v20 = sub_2B3D560(v57 + 80, v69.m128i_i64), v11 = v57, v10 = a5, v20) )
LABEL_3:
        v12 = (_BYTE *)a3;
    }
    v56 = v10;
    v68 = 257;
    v13 = *(_QWORD *)(v11 + 3344);
    v60 = v11;
    v76 = 257;
    v69 = (__m128i)v13;
    v70 = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    v14 = sub_9AC470(a3, &v69, 0);
    v5 = sub_921630((unsigned int **)(v60 + 3368), (__int64)v12, v56, v14 ^ 1u, (__int64)v67);
    v9 = *(_QWORD *)(v5 + 8);
  }
  v15 = *a1;
  if ( *(_BYTE *)(v9 + 8) == 17 )
  {
    v16 = (_BYTE *)sub_2B330C0(*a1 + 3368, a2, v5, a4 * *(_DWORD *)(v9 + 32), 0, 0);
    v17 = (__int64)v16;
    if ( *v16 != 85 )
      return v17;
    v39 = *((_QWORD *)v16 - 4);
    if ( !v39
      || *(_BYTE *)v39
      || *(_QWORD *)(v39 + 24) != *(_QWORD *)(v17 + 80)
      || (*(_BYTE *)(v39 + 33) & 0x20) == 0
      || *(_DWORD *)(v39 + 36) != 382 )
    {
      return v17;
    }
  }
  else
  {
    v55 = *a1 + 3368;
    v68 = 257;
    v61 = v15;
    v21 = sub_BCB2D0(*(_QWORD **)(v15 + 3440));
    v22 = sub_ACD640(v21, a4, 0);
    v23 = v61;
    v24 = *(_QWORD *)(v61 + 3448);
    v62 = v22;
    v58 = v23;
    v17 = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64))(*(_QWORD *)v24 + 104LL))(v24, a2, v5, v22);
    if ( !v17 )
    {
      LOWORD(v72) = 257;
      v42 = sub_BD2C40(72, 3u);
      v43 = v55;
      v44 = v58;
      v17 = (__int64)v42;
      if ( v42 )
      {
        sub_B4DFA0((__int64)v42, (__int64)a2, v5, v62, (__int64)&v69, v58, 0, 0);
        v44 = v58;
        v43 = v55;
      }
      v63 = v44;
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v44 + 3456) + 16LL))(
        *(_QWORD *)(v44 + 3456),
        v17,
        v67,
        *(_QWORD *)(v43 + 56),
        *(_QWORD *)(v43 + 64));
      v45 = *(_QWORD *)(v63 + 3368);
      v46 = v45;
      v64 = v45 + 16LL * *(unsigned int *)(v63 + 3376);
      if ( v45 != v64 )
      {
        do
        {
          v47 = *(_QWORD *)(v46 + 8);
          v48 = *(_DWORD *)v46;
          v46 += 16;
          sub_B99FD0(v17, v48, v47);
        }
        while ( v64 != v46 );
      }
    }
    if ( *(_BYTE *)v17 != 91 )
      return v17;
  }
  v25 = *a1;
  v66 = v17;
  sub_2400480((__int64)&v69, v25 + 3112, &v66);
  if ( (_BYTE)v72 )
  {
    v40 = *(unsigned int *)(v25 + 3152);
    v41 = v66;
    if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v25 + 3156) )
    {
      v65 = v66;
      sub_C8D5F0(v25 + 3144, (const void *)(v25 + 3160), v40 + 1, 8u, (__int64)&v69, v66);
      v40 = *(unsigned int *)(v25 + 3152);
      v41 = v65;
    }
    *(_QWORD *)(*(_QWORD *)(v25 + 3144) + 8 * v40) = v41;
    ++*(_DWORD *)(v25 + 3152);
  }
  v26 = *a1 + 3160;
  v67[0] = *(_QWORD *)(v66 + 40);
  sub_29B09C0((__int64)&v69, v26, v67);
  if ( *(_BYTE *)a3 > 0x1Cu )
  {
    v27 = *a1;
    v28 = (__int64 *)sub_2B2A0E0(v27, a3);
    if ( v29 )
    {
      if ( v5 == a3 )
      {
        v5 = v66;
        if ( !v66 )
          return v17;
      }
      else if ( *(_BYTE *)v5 <= 0x1Cu )
      {
        return v17;
      }
      v30 = *v28;
      v31 = sub_2B2A000(*v28, a3);
      v32 = *(unsigned int *)(v27 + 2244);
      v33 = *(_QWORD *)(v27 + 2232);
      v34 = v31;
      v35 = *(unsigned int *)(v27 + 2240);
      v36 = &v69;
      v37 = *(_DWORD *)(v27 + 2240);
      if ( v35 >= v32 )
      {
        v49 = v35 + 1;
        v69.m128i_i64[0] = a3;
        v69.m128i_i64[1] = v5;
        v70 = v30;
        LODWORD(v71) = v34;
        if ( v32 < v35 + 1 )
        {
          v52 = v27 + 2232;
          v53 = (const void *)(v27 + 2248);
          if ( v33 > (unsigned __int64)&v69 || (v59 = v33, (unsigned __int64)&v69 >= v33 + 32 * v35) )
          {
            sub_C8D5F0(v52, v53, v49, 0x20u, (__int64)&v69, v33);
            v35 = *(unsigned int *)(v27 + 2240);
            v36 = &v69;
            v50 = *(_QWORD *)(v27 + 2232);
          }
          else
          {
            sub_C8D5F0(v52, v53, v49, 0x20u, (__int64)&v69, v33);
            v50 = *(_QWORD *)(v27 + 2232);
            v35 = *(unsigned int *)(v27 + 2240);
            v36 = (__m128i *)((char *)&v69 + v50 - v59);
          }
        }
        else
        {
          v50 = *(_QWORD *)(v27 + 2232);
        }
        v51 = (__m128i *)(v50 + 32 * v35);
        *v51 = _mm_loadu_si128(v36);
        v51[1] = _mm_loadu_si128(v36 + 1);
        ++*(_DWORD *)(v27 + 2240);
      }
      else
      {
        v38 = v33 + 32 * v35;
        if ( v38 )
        {
          *(_QWORD *)v38 = a3;
          *(_QWORD *)(v38 + 8) = v5;
          *(_QWORD *)(v38 + 16) = v30;
          *(_DWORD *)(v38 + 24) = v34;
          v37 = *(_DWORD *)(v27 + 2240);
        }
        *(_DWORD *)(v27 + 2240) = v37 + 1;
      }
    }
  }
  return v17;
}
