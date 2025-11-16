// Function: sub_2D02FE0
// Address: 0x2d02fe0
//
__int64 __fastcall sub_2D02FE0(__int64 a1, __int64 *a2)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  _BYTE *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rbx
  const void *v9; // r12
  size_t v10; // rdx
  size_t v11; // rbx
  int v12; // eax
  unsigned int v13; // r15d
  __int64 *v14; // r10
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  size_t v23; // r15
  int v24; // eax
  unsigned int v25; // r9d
  __int64 *v26; // r10
  __int64 v27; // rdx
  _BYTE *v28; // rsi
  __int64 v29; // rax
  unsigned int v30; // r9d
  __int64 *v31; // r10
  __int64 v32; // rcx
  __int64 *v33; // rax
  __int64 *v34; // rax
  int v35; // ebx
  __int64 **v36; // r13
  __int64 (__fastcall ***v37)(); // rdi
  __int64 (__fastcall **v38)(); // rsi
  _BYTE *v39; // rsi
  unsigned int v40; // r12d
  _BYTE *v41; // rsi
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // r10
  __int64 v46; // rcx
  __int64 *v47; // rax
  __int64 *v48; // rax
  __int64 v50; // [rsp+18h] [rbp-338h]
  __int64 v51; // [rsp+28h] [rbp-328h]
  __int64 v52; // [rsp+30h] [rbp-320h]
  __int64 v53; // [rsp+58h] [rbp-2F8h]
  __int64 *v54; // [rsp+60h] [rbp-2F0h]
  unsigned int v55; // [rsp+78h] [rbp-2D8h]
  int v56; // [rsp+7Ch] [rbp-2D4h]
  __int64 v57; // [rsp+C8h] [rbp-288h]
  int src; // [rsp+D0h] [rbp-280h]
  void *srca; // [rsp+D0h] [rbp-280h]
  __int64 *v60; // [rsp+D8h] [rbp-278h]
  __int64 v61; // [rsp+E0h] [rbp-270h]
  int v62; // [rsp+E8h] [rbp-268h]
  __int64 v63; // [rsp+E8h] [rbp-268h]
  unsigned __int8 v64; // [rsp+E8h] [rbp-268h]
  _BYTE *v65; // [rsp+F0h] [rbp-260h] BYREF
  __int64 v66; // [rsp+F8h] [rbp-258h]
  _BYTE v67[16]; // [rsp+100h] [rbp-250h] BYREF
  unsigned __int64 v68[2]; // [rsp+110h] [rbp-240h] BYREF
  _BYTE v69[32]; // [rsp+120h] [rbp-230h] BYREF
  __m128i dest; // [rsp+140h] [rbp-210h] BYREF
  _QWORD v71[8]; // [rsp+150h] [rbp-200h] BYREF
  __m128i v72; // [rsp+190h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v73)(); // [rsp+1A0h] [rbp-1B0h] BYREF
  _QWORD v74[7]; // [rsp+1A8h] [rbp-1A8h] BYREF
  volatile signed __int32 *v75; // [rsp+1E0h] [rbp-170h] BYREF
  int v76; // [rsp+1E8h] [rbp-168h]
  _QWORD *v77; // [rsp+1F0h] [rbp-160h] BYREF
  _QWORD v78[2]; // [rsp+200h] [rbp-150h] BYREF
  _QWORD v79[28]; // [rsp+210h] [rbp-140h] BYREF
  __int16 v80; // [rsp+2F0h] [rbp-60h]
  __int64 v81; // [rsp+2F8h] [rbp-58h]
  __int64 v82; // [rsp+300h] [rbp-50h]
  __int64 v83; // [rsp+308h] [rbp-48h]
  __int64 v84; // [rsp+310h] [rbp-40h]

  v61 = sub_BA8DC0((__int64)a2, (__int64)"nvvm.reflection", 15);
  if ( v61 )
  {
    src = sub_B91A00(v61);
    if ( src )
    {
      v3 = 0;
      while ( 1 )
      {
        v16 = sub_B91A10(v61, v3);
        v17 = *(_BYTE *)(v16 - 16);
        if ( (v17 & 2) != 0 )
        {
          v4 = *(_QWORD *)(v16 - 32);
          v5 = *(_BYTE **)v4;
          if ( **(_BYTE **)v4 )
            v5 = 0;
        }
        else
        {
          v4 = -16 - 8LL * ((v17 >> 2) & 0xF) + v16;
          v5 = *(_BYTE **)v4;
          if ( **(_BYTE **)v4 )
            v5 = 0;
        }
        v6 = *(_QWORD *)(v4 + 8);
        if ( *(_BYTE *)v6 != 1 || (v7 = *(_QWORD *)(v6 + 136), *(_BYTE *)v7 != 17) )
          BUG();
        v8 = *(_QWORD **)(v7 + 24);
        if ( *(_DWORD *)(v7 + 32) > 0x40u )
          v8 = (_QWORD *)*v8;
        v62 = (int)v8;
        v9 = (const void *)sub_B91420((__int64)v5);
        v11 = v10;
        v12 = sub_C92610();
        v13 = sub_C92740(a1, v9, v11, v12);
        v14 = (__int64 *)(*(_QWORD *)a1 + 8LL * v13);
        v15 = *v14;
        if ( !*v14 )
          goto LABEL_70;
        if ( v15 == -8 )
          break;
LABEL_12:
        ++v3;
        *(_DWORD *)(v15 + 8) = v62;
        if ( src == v3 )
          goto LABEL_17;
      }
      --*(_DWORD *)(a1 + 16);
LABEL_70:
      v60 = v14;
      v44 = sub_C7D670(v11 + 17, 8);
      v45 = v60;
      v46 = v44;
      if ( v11 )
      {
        v57 = v44;
        memcpy((void *)(v44 + 16), v9, v11);
        v45 = v60;
        v46 = v57;
      }
      *(_BYTE *)(v46 + v11 + 16) = 0;
      *(_QWORD *)v46 = v11;
      *(_DWORD *)(v46 + 8) = 0;
      *v45 = v46;
      ++*(_DWORD *)(a1 + 12);
      v47 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v13));
      v15 = *v47;
      if ( !*v47 || v15 == -8 )
      {
        v48 = v47 + 1;
        do
        {
          do
            v15 = *v48++;
          while ( !v15 );
        }
        while ( v15 == -8 );
      }
      goto LABEL_12;
    }
  }
LABEL_17:
  v18 = qword_5014CE8;
  v19 = (qword_5014CF0 - qword_5014CE8) >> 5;
  if ( (_DWORD)v19 )
  {
    v51 = 0;
    v50 = 32LL * (unsigned int)(v19 - 1);
    while ( 1 )
    {
      dest.m128i_i64[0] = (__int64)v71;
      dest.m128i_i64[1] = 0x400000000LL;
      v72 = *(__m128i *)(v51 + v18);
      sub_C937F0(&v72, (__int64)&dest, ",", 1u, 0xFFFFFFFFLL, 1);
      if ( dest.m128i_i32[2] )
        break;
LABEL_42:
      if ( (_QWORD *)dest.m128i_i64[0] != v71 )
        _libc_free(dest.m128i_u64[0]);
      if ( v50 == v51 )
        goto LABEL_46;
      v18 = qword_5014CE8;
      v51 += 32;
    }
    v63 = 0;
    v53 = 16LL * dest.m128i_u32[2];
    while ( 1 )
    {
      v68[0] = (unsigned __int64)v69;
      v68[1] = 0x200000000LL;
      sub_C937F0((const __m128i *)(dest.m128i_i64[0] + v63), (__int64)v68, "=", 1u, 0xFFFFFFFFLL, 1);
      v28 = *(_BYTE **)(v68[0] + 16);
      if ( v28 )
      {
        v20 = (__int64)&v28[*(_QWORD *)(v68[0] + 24)];
        v65 = v67;
        sub_2D02680((__int64 *)&v65, v28, v20);
      }
      else
      {
        v67[0] = 0;
        v66 = 0;
        v65 = v67;
      }
      sub_222DF20((__int64)v79);
      v72.m128i_i64[0] = (__int64)qword_4A072D8;
      v79[27] = 0;
      v79[0] = off_4A06798;
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      v84 = 0;
      *(__int64 *)((char *)v72.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
      v72.m128i_i64[1] = 0;
      sub_222DD70((__int64)v72.m128i_i64 + *(_QWORD *)(v72.m128i_i64[0] - 24), 0);
      v73 = (__int64 (__fastcall **)())qword_4A07288;
      *(_QWORD *)((char *)&v74[-1] + qword_4A07288[-3]) = &unk_4A072B0;
      sub_222DD70((__int64)&v74[-1] + (_QWORD)*(v73 - 3), 0);
      v72.m128i_i64[0] = (__int64)qword_4A07328;
      *(__int64 *)((char *)v72.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
      memset(&v74[1], 0, 48);
      v72.m128i_i64[0] = (__int64)off_4A073F0;
      v79[0] = off_4A07440;
      v73 = off_4A07418;
      v74[0] = off_4A07480;
      sub_220A990(&v75);
      v76 = 0;
      v74[0] = off_4A07080;
      v77 = v78;
      sub_2D02680((__int64 *)&v77, v65, (__int64)&v65[v66]);
      v76 = 24;
      sub_223FD50((__int64)v74, (__int64)v77, 0, 0);
      sub_222DD70((__int64)v79, (__int64)v74);
      if ( v65 != v67 )
        j_j___libc_free_0((unsigned __int64)v65);
      sub_222E4D0(v72.m128i_i64, &v65, v21, v22);
      v56 = (int)v65;
      v23 = *(_QWORD *)(v68[0] + 8);
      srca = *(void **)v68[0];
      v24 = sub_C92610();
      v25 = sub_C92740(a1, srca, v23, v24);
      v26 = (__int64 *)(*(_QWORD *)a1 + 8LL * v25);
      v27 = *v26;
      if ( !*v26 )
        goto LABEL_34;
      if ( v27 == -8 )
        break;
LABEL_26:
      *(_DWORD *)(v27 + 8) = v56;
      v72.m128i_i64[0] = (__int64)off_4A073F0;
      v79[0] = off_4A07440;
      v73 = off_4A07418;
      v74[0] = off_4A07080;
      if ( v77 != v78 )
        j_j___libc_free_0((unsigned __int64)v77);
      v74[0] = off_4A07480;
      sub_2209150(&v75);
      v72.m128i_i64[0] = (__int64)qword_4A07328;
      *(__int64 *)((char *)v72.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
      v73 = (__int64 (__fastcall **)())qword_4A07288;
      *(_QWORD *)((char *)&v74[-1] + qword_4A07288[-3]) = &unk_4A072B0;
      v72.m128i_i64[0] = (__int64)qword_4A072D8;
      *(__int64 *)((char *)v72.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
      v72.m128i_i64[1] = 0;
      v79[0] = off_4A06798;
      sub_222E050((__int64)v79);
      if ( (_BYTE *)v68[0] != v69 )
        _libc_free(v68[0]);
      v63 += 16;
      if ( v53 == v63 )
        goto LABEL_42;
    }
    --*(_DWORD *)(a1 + 16);
LABEL_34:
    v54 = v26;
    v55 = v25;
    v29 = sub_C7D670(v23 + 17, 8);
    v30 = v55;
    v31 = v54;
    v32 = v29;
    if ( v23 )
    {
      v52 = v29;
      memcpy((void *)(v29 + 16), srca, v23);
      v30 = v55;
      v31 = v54;
      v32 = v52;
    }
    *(_BYTE *)(v32 + v23 + 16) = 0;
    *(_QWORD *)v32 = v23;
    *(_DWORD *)(v32 + 8) = 0;
    *v31 = v32;
    ++*(_DWORD *)(a1 + 12);
    v33 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v30));
    v27 = *v33;
    if ( *v33 == -8 || !v27 )
    {
      v34 = v33 + 1;
      do
      {
        do
          v27 = *v34++;
        while ( v27 == -8 );
      }
      while ( !v27 );
    }
    goto LABEL_26;
  }
LABEL_46:
  v35 = 0;
  dest.m128i_i64[1] = 0;
  dest.m128i_i64[0] = (__int64)v71;
  LOBYTE(v71[0]) = 0;
  v64 = 0;
  v36 = (__int64 **)sub_BCB2B0((_QWORD *)*a2);
  do
  {
    v68[0] = sub_BCE760(v36, v35);
    sub_B6E0E0(&v72, 0x24B9u, (__int64)v68, 1, a2, 0);
    v37 = (__int64 (__fastcall ***)())dest.m128i_i64[0];
    if ( (__int64 (__fastcall ***)())v72.m128i_i64[0] == &v73 )
    {
      v43 = v72.m128i_i64[1];
      if ( v72.m128i_i64[1] )
      {
        if ( v72.m128i_i64[1] == 1 )
          *(_BYTE *)dest.m128i_i64[0] = (_BYTE)v73;
        else
          memcpy((void *)dest.m128i_i64[0], &v73, v72.m128i_u64[1]);
        v43 = v72.m128i_i64[1];
        v37 = (__int64 (__fastcall ***)())dest.m128i_i64[0];
      }
      dest.m128i_i64[1] = v43;
      *((_BYTE *)v37 + v43) = 0;
      v37 = (__int64 (__fastcall ***)())v72.m128i_i64[0];
    }
    else
    {
      if ( (_QWORD *)dest.m128i_i64[0] == v71 )
      {
        dest = v72;
        v71[0] = v73;
      }
      else
      {
        v38 = (__int64 (__fastcall **)())v71[0];
        dest = v72;
        v71[0] = v73;
        if ( v37 )
        {
          v72.m128i_i64[0] = (__int64)v37;
          v73 = v38;
          goto LABEL_51;
        }
      }
      v37 = &v73;
      v72.m128i_i64[0] = (__int64)&v73;
    }
LABEL_51:
    v72.m128i_i64[1] = 0;
    *(_BYTE *)v37 = 0;
    if ( (__int64 (__fastcall ***)())v72.m128i_i64[0] != &v73 )
      j_j___libc_free_0(v72.m128i_u64[0]);
    v39 = sub_BA8CB0((__int64)a2, dest.m128i_i64[0], dest.m128i_u64[1]);
    if ( v39 )
      v64 |= sub_2D02CA0(a1, (__int64)v39);
    ++v35;
  }
  while ( v35 != 5 );
  v40 = v64;
  v41 = sub_BA8CB0((__int64)a2, (__int64)"__nvvm_reflect", 0xEu);
  if ( v41 )
    v40 = sub_2D02CA0(a1, (__int64)v41) | v64;
  if ( (_QWORD *)dest.m128i_i64[0] != v71 )
    j_j___libc_free_0(dest.m128i_u64[0]);
  return v40;
}
