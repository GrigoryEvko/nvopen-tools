// Function: sub_30FD220
// Address: 0x30fd220
//
__int64 *__fastcall sub_30FD220(__int64 *a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // edx
  unsigned __int8 *v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rsi
  _QWORD *v14; // rdx
  char v15; // r14
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // r15d
  unsigned __int64 *v24; // r15
  unsigned __int64 *v25; // r14
  __int64 v26; // r8
  unsigned __int64 *v27; // r15
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r14
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // esi
  bool v38; // al
  __int64 i; // rax
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rax
  __m128i *v43; // rax
  __int64 v44; // rsi
  unsigned __int64 *v45; // r14
  unsigned __int64 *v46; // rsi
  __m128i *v47; // rbx
  unsigned __int64 *v48; // r12
  __int64 v49; // r8
  unsigned __int64 v50; // rdi
  unsigned __int8 *v51; // [rsp+0h] [rbp-3D0h]
  int v52; // [rsp+0h] [rbp-3D0h]
  __int64 v53; // [rsp+0h] [rbp-3D0h]
  __int64 v54; // [rsp+8h] [rbp-3C8h]
  unsigned __int8 *v55; // [rsp+8h] [rbp-3C8h]
  __int64 v56; // [rsp+8h] [rbp-3C8h]
  unsigned __int8 *v57; // [rsp+8h] [rbp-3C8h]
  _QWORD *v58; // [rsp+8h] [rbp-3C8h]
  unsigned __int8 *v59; // [rsp+8h] [rbp-3C8h]
  __int64 v60; // [rsp+10h] [rbp-3C0h]
  __int32 v61; // [rsp+10h] [rbp-3C0h]
  __int64 v62; // [rsp+10h] [rbp-3C0h]
  int v63; // [rsp+18h] [rbp-3B8h]
  __int64 *v64; // [rsp+18h] [rbp-3B8h]
  __int64 v65; // [rsp+18h] [rbp-3B8h]
  _QWORD *v66; // [rsp+18h] [rbp-3B8h]
  __int64 *v67; // [rsp+20h] [rbp-3B0h]
  __int64 v68; // [rsp+28h] [rbp-3A8h]
  __int64 v69; // [rsp+38h] [rbp-398h] BYREF
  void *v70; // [rsp+40h] [rbp-390h] BYREF
  __int32 v71; // [rsp+48h] [rbp-388h]
  __int8 v72; // [rsp+4Ch] [rbp-384h]
  __int64 v73; // [rsp+50h] [rbp-380h]
  __m128i v74; // [rsp+58h] [rbp-378h]
  __int64 v75; // [rsp+68h] [rbp-368h]
  __m128i v76; // [rsp+70h] [rbp-360h]
  __m128i v77; // [rsp+80h] [rbp-350h]
  __m128i *v78; // [rsp+90h] [rbp-340h] BYREF
  __int64 v79; // [rsp+98h] [rbp-338h]
  _BYTE v80[320]; // [rsp+A0h] [rbp-330h] BYREF
  char v81; // [rsp+1E0h] [rbp-1F0h]
  int v82; // [rsp+1E4h] [rbp-1ECh]
  __int64 v83; // [rsp+1E8h] [rbp-1E8h]
  __m128i v84; // [rsp+1F0h] [rbp-1E0h] BYREF
  __int64 v85; // [rsp+200h] [rbp-1D0h]
  __m128i v86; // [rsp+208h] [rbp-1C8h] BYREF
  __int64 v87; // [rsp+218h] [rbp-1B8h]
  __m128i v88; // [rsp+220h] [rbp-1B0h] BYREF
  __m128i v89; // [rsp+230h] [rbp-1A0h] BYREF
  unsigned __int64 *v90; // [rsp+240h] [rbp-190h]
  unsigned int v91; // [rsp+248h] [rbp-188h]
  char v92; // [rsp+250h] [rbp-180h] BYREF
  char v93; // [rsp+254h] [rbp-17Ch]
  char v94; // [rsp+390h] [rbp-40h]
  int v95; // [rsp+394h] [rbp-3Ch]
  __int64 v96; // [rsp+398h] [rbp-38h]

  v4 = a3;
  v5 = a2;
  sub_30FBAE0(v84.m128i_i64, a2, a3);
  v6 = v84.m128i_i64[0];
  if ( v84.m128i_i64[0] )
  {
LABEL_2:
    *a1 = v6;
    return a1;
  }
  v68 = sub_B491C0((__int64)v4);
  v8 = *((_QWORD *)v4 - 4);
  if ( v8 && !*(_BYTE *)v8 && *((_QWORD *)v4 + 10) == *(_QWORD *)(v8 + 24) )
    v6 = *((_QWORD *)v4 - 4);
  v9 = *(_QWORD *)(a2 + 16);
  v69 = a2;
  v60 = sub_BC1CD0(v9, &unk_4F89C30, v6);
  v54 = sub_BC1CD0(*(_QWORD *)(a2 + 16), &unk_4F8FAE8, v68);
  v67 = (__int64 *)(v54 + 8);
  if ( dword_5031908 == 1 )
  {
    v12 = *(_QWORD *)(a2 + 368);
    v13 = v68;
    if ( !sub_D84460(v12, v68) )
    {
      if ( *(_QWORD *)(v5 + 104) )
      {
        v15 = (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(v5 + 112))(v5 + 88, v4);
LABEL_22:
        v16 = sub_22077B0(0x40u);
        v6 = v16;
        if ( v16 )
          sub_30CABE0(v16, v5, v4, (__int64)v67, v15);
        goto LABEL_2;
      }
LABEL_83:
      sub_4263D6(v12, v13, v14);
    }
  }
  v63 = sub_30CC550((__int64)v4, *(_QWORD *)(v5 + 16));
  if ( v63 == 2 || v68 == v6 )
  {
    (*(void (__fastcall **)(__int64 *, __int64, unsigned __int8 *, _QWORD))(*(_QWORD *)v5 + 56LL))(a1, v5, v4, 0);
    return a1;
  }
  if ( *(_BYTE *)(v5 + 360) )
  {
    v17 = *(_QWORD *)(v54 + 8);
    v18 = sub_B2BE50(v17);
    if ( sub_B6EA50(v18)
      || (v41 = sub_B2BE50(v17),
          v42 = sub_B6F970(v41),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v42 + 48LL))(v42)) )
    {
      sub_B176B0((__int64)&v84, (__int64)"inline-ml", (__int64)"ForceStop", 9, (__int64)v4);
      sub_B18290((__int64)&v84, "Won't attempt inlining because module size grew too much.", 0x39u);
      v71 = v84.m128i_i32[2];
      v23 = v91;
      v74 = _mm_loadu_si128(&v86);
      v72 = v84.m128i_i8[12];
      v76 = _mm_loadu_si128(&v88);
      v73 = v85;
      v77 = _mm_loadu_si128(&v89);
      v70 = &unk_49D9D40;
      v75 = v87;
      v78 = (__m128i *)v80;
      v79 = 0x400000000LL;
      if ( v91 )
      {
        v43 = (__m128i *)v80;
        v44 = v91;
        if ( v91 > 4 )
        {
          sub_11F02D0((__int64)&v78, v91, v19, v20, v21, v22);
          v43 = v78;
          v44 = v91;
        }
        v45 = v90;
        v46 = &v90[10 * v44];
        if ( v90 != v46 )
        {
          v62 = v5;
          v47 = v43;
          v59 = v4;
          v48 = v90;
          do
          {
            if ( v47 )
            {
              v47->m128i_i64[0] = (__int64)v47[1].m128i_i64;
              sub_30FA730(v47->m128i_i64, (_BYTE *)*v48, *v48 + v48[1]);
              v47[2].m128i_i64[0] = (__int64)v47[3].m128i_i64;
              sub_30FA730(v47[2].m128i_i64, (_BYTE *)v48[4], v48[4] + v48[5]);
              v47[4] = _mm_loadu_si128((const __m128i *)v48 + 4);
            }
            v48 += 10;
            v47 += 5;
          }
          while ( v46 != v48 );
          v5 = v62;
          v4 = v59;
          v45 = v90;
        }
        LODWORD(v79) = v23;
        v81 = v94;
        v82 = v95;
        v83 = v96;
        v70 = &unk_49D9DB0;
        v84.m128i_i64[0] = (__int64)&unk_49D9D40;
        v49 = 10LL * v91;
        v24 = &v45[v49];
        if ( v45 != &v45[v49] )
        {
          do
          {
            v24 -= 10;
            v50 = v24[4];
            if ( (unsigned __int64 *)v50 != v24 + 6 )
              j_j___libc_free_0(v50);
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              j_j___libc_free_0(*v24);
          }
          while ( v45 != v24 );
          v24 = v90;
        }
      }
      else
      {
        v24 = v90;
        v81 = v94;
        v82 = v95;
        v83 = v96;
        v70 = &unk_49D9DB0;
      }
      if ( v24 != (unsigned __int64 *)&v92 )
        _libc_free((unsigned __int64)v24);
      sub_1049740(v67, (__int64)&v70);
      v25 = (unsigned __int64 *)v78;
      v70 = &unk_49D9D40;
      v26 = 5LL * (unsigned int)v79;
      v27 = (unsigned __int64 *)&v78[v26];
      if ( v78 != &v78[v26] )
      {
        do
        {
          v27 -= 10;
          v28 = v27[4];
          if ( (unsigned __int64 *)v28 != v27 + 6 )
            j_j___libc_free_0(v28);
          if ( (unsigned __int64 *)*v27 != v27 + 2 )
            j_j___libc_free_0(*v27);
        }
        while ( v25 != v27 );
        v27 = (unsigned __int64 *)v78;
      }
      if ( v27 != (unsigned __int64 *)v80 )
        _libc_free((unsigned __int64)v27);
    }
    v15 = v63 == 1;
    goto LABEL_22;
  }
  if ( v63 == 1 )
  {
    sub_30E0710(&v84, (__int64)v4, (__int64 *)(v60 + 8), (__int64)sub_30FA650, (__int64)&v69, 0, 0, 0, 0, 0);
    if ( v93 )
    {
      (*(void (__fastcall **)(__int64 *, __int64, unsigned __int8 *, __int64))(*(_QWORD *)v5 + 56LL))(a1, v5, v4, 1);
      return a1;
    }
  }
  else
  {
    v64 = (__int64 *)(v60 + 8);
    v84.m128i_i64[0] = sub_30E01C0((__int64)v4, v60 + 8, (__int64)sub_30FA650, (__int64)&v69, 0, 0, 0, 0, 0, 0);
    if ( v84.m128i_i8[4] )
    {
      v61 = v84.m128i_i32[0];
      sub_30E0710(&v84, (__int64)v4, v64, (__int64)sub_30FA650, (__int64)&v69, 0, 0, 0, 0, 0);
      if ( v93 )
      {
        v10 = *v4;
        v11 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        if ( v10 == 40 )
        {
          v55 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
          v31 = sub_B491D0((__int64)v4);
          v11 = v55;
          v65 = -32 - 32LL * v31;
        }
        else
        {
          v65 = -32;
          if ( v10 != 85 )
          {
            v65 = -96;
            if ( v10 != 34 )
LABEL_84:
              BUG();
          }
        }
        if ( (v4[7] & 0x80u) != 0 )
        {
          v51 = v11;
          v32 = sub_BD2BC0((__int64)v4);
          v11 = v51;
          v56 = v33 + v32;
          if ( (v4[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v56 >> 4) )
              goto LABEL_84;
          }
          else
          {
            v34 = sub_BD2BC0((__int64)v4);
            v11 = v51;
            if ( (unsigned int)((v56 - v34) >> 4) )
            {
              v57 = v51;
              if ( (v4[7] & 0x80u) == 0 )
                goto LABEL_84;
              v52 = *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8);
              if ( (v4[7] & 0x80u) == 0 )
                BUG();
              v35 = sub_BD2BC0((__int64)v4);
              v11 = v57;
              v65 -= 32LL * (unsigned int)(*(_DWORD *)(v35 + v36 - 4) - v52);
            }
          }
        }
        if ( v11 == &v4[v65] )
        {
          v53 = 0;
        }
        else
        {
          v37 = 0;
          do
          {
            v38 = **(_BYTE **)v11 <= 0x15u;
            v11 += 32;
            v37 += v38;
          }
          while ( v11 != &v4[v65] );
          v53 = v37;
        }
        v58 = sub_30FCBF0(v5, v68);
        v12 = v5;
        v66 = sub_30FCBF0(v5, v6);
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 200LL) = *v66;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 208LL) = (unsigned int)sub_30FBA00((_QWORD *)v5, v68);
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 216LL) = *(_QWORD *)(v5 + 176);
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 224LL) = v53;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 240LL) = *(_QWORD *)(v5 + 184);
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 248LL) = v58[2];
        v13 = v58[1];
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 256LL) = v13;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 264LL) = *v58;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 272LL) = v66[1];
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 280LL) = v66[2];
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 232LL) = v61;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 288LL) = (*(_BYTE *)(v6 + 32) & 0xF) == 1;
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 296LL) = (*(_BYTE *)(v68 + 32) & 0xF) == 1;
        for ( i = 0; i != 25; ++i )
        {
          v40 = v84.m128i_i32[i];
          v14 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 8 * i);
          *v14 = v40;
        }
        if ( qword_5031C70 && (_BYTE)qword_5031B68 )
        {
          if ( !*(_QWORD *)(v5 + 104) )
            goto LABEL_83;
          **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v5 + 80) + 24LL) + 200LL) = (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 *))(v5 + 112))(
                                                                              v5 + 88,
                                                                              v4);
        }
        (*(void (__fastcall **)(void **, __int64, unsigned __int8 *, __int64 *))(*(_QWORD *)v5 + 72LL))(
          &v70,
          v5,
          v4,
          v67);
        *a1 = (__int64)v70;
        return a1;
      }
    }
  }
  v29 = sub_22077B0(0x40u);
  v30 = v29;
  if ( v29 )
    sub_30CABE0(v29, v5, v4, (__int64)v67, 0);
  *a1 = v30;
  return a1;
}
