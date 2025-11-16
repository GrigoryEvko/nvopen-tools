// Function: sub_8B74F0
// Address: 0x8b74f0
//
_QWORD *__fastcall sub_8B74F0(unsigned __int64 a1, __int64 ***a2, unsigned int a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __m128i *v7; // rbx
  _QWORD *v8; // rax
  unsigned __int64 v9; // r12
  char v10; // al
  __int64 **v11; // r14
  char v12; // al
  __int64 *v13; // rdi
  unsigned __int8 *v14; // rdi
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  _QWORD *v21; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r15
  __m128i *v32; // r14
  _QWORD *v33; // rax
  __m128i *v34; // rax
  __int64 *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 **v45; // rax
  __int64 *v46; // rdx
  __int64 v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+18h] [rbp-68h]
  int v52; // [rsp+24h] [rbp-5Ch] BYREF
  int v53; // [rsp+28h] [rbp-58h] BYREF
  int v54; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 *v55[2]; // [rsp+30h] [rbp-50h] BYREF
  int v56; // [rsp+40h] [rbp-40h]

  v9 = a1;
  v10 = *(_BYTE *)(a1 + 80);
  v11 = *a2;
  if ( v10 == 16 )
  {
    v9 = **(_QWORD **)(a1 + 88);
    v10 = *(_BYTE *)(v9 + 80);
  }
  if ( v10 == 24 )
  {
    v9 = *(_QWORD *)(v9 + 88);
    v10 = *(_BYTE *)(v9 + 80);
  }
  if ( v10 == 10 )
    v48 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
  else
    v48 = *(_QWORD *)(v9 + 88);
  if ( v11 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *((_BYTE *)v11 + 8);
        if ( !v12 )
          break;
        if ( v12 == 1 )
        {
          if ( (unsigned int)sub_8D3A70(v11[4][16]) && !(unsigned int)sub_730B80((__int64)v11[4]) )
            sub_685360(0xA55u, a4, v11[4][16]);
          if ( sub_72A990((__int64)v11[4]) )
          {
            sub_6851C0(0x1DBu, a4);
            sub_72C970((__int64)v11[4]);
          }
        }
        v11 = (__int64 **)*v11;
        if ( !v11 )
        {
LABEL_19:
          v11 = *a2;
          goto LABEL_20;
        }
      }
      if ( (unsigned int)sub_8DD0E0(v11[4], &v52, &v53, &v54, v55) )
      {
        if ( v53 )
        {
          sub_6851C0(0x1D4u, a4);
          v24 = sub_72C930();
          v11[4] = (__int64 *)v24;
          v13 = (__int64 *)v24;
          goto LABEL_18;
        }
        if ( v54 )
        {
          sub_6851C0(0x67Cu, a4);
          v25 = sub_72C930();
          v11[4] = (__int64 *)v25;
          v13 = (__int64 *)v25;
          goto LABEL_18;
        }
        if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu )
          sub_6851C0(v52 == 0 ? 1658 : 510, a4);
      }
      v13 = v11[4];
LABEL_18:
      v11[4] = (__int64 *)sub_8E3240(v13, 0);
      v11 = (__int64 **)*v11;
      if ( !v11 )
        goto LABEL_19;
    }
  }
LABEL_20:
  v55[0] = (__int64 *)v9;
  v55[1] = (__int64 *)v11;
  v56 = 0;
  v14 = *(unsigned __int8 **)(v48 + 136);
  if ( !v14 )
  {
    v14 = (unsigned __int8 *)sub_881A70(0, 0xBu, 12, 13, a5, a6);
    *(_QWORD *)(v48 + 136) = v14;
  }
  v15 = (__int64 *)sub_881B20(v14, (__int64)v55, 0);
  if ( v15 )
  {
    if ( *v15 )
    {
      v20 = sub_892240(*v15);
      if ( v20 )
      {
        v21 = *(_QWORD **)(v20 + 24);
        sub_88FC00(v21[11], *a2, a3);
LABEL_26:
        sub_725130((__int64 *)*a2);
        goto LABEL_27;
      }
    }
  }
  v23 = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 176LL);
  if ( (unsigned int)sub_893F30((__int64 *)*a2, (__int64)v55, v16, v17, v18, v19)
    || (unsigned int)sub_8D97B0(*(_QWORD *)(v23 + 152))
    || (v26 = *(_QWORD **)(v48 + 104), (v27 = v26[22]) != 0)
    && (*(_QWORD *)(v27 + 16) || (*(_BYTE *)(*(_QWORD *)(*v26 + 88LL) + 160LL) & 0x20) != 0)
    && !(unsigned int)sub_89A370((__int64 *)*a2)
    && !(unsigned int)sub_8A00C0(
                        v9,
                        (__int64 *)*a2,
                        ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) >> 4) ^ 1) & 1) )
  {
    if ( (*(_BYTE *)(v9 + 81) & 0x10) != 0 )
      v51 = *(_QWORD *)(v9 + 64);
    else
      v51 = 0;
    sub_7296C0(&v54);
    switch ( *(_BYTE *)(v9 + 80) )
    {
      case 4:
      case 5:
        v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
        v30 = *(_QWORD *)(v9 + 88);
        break;
      case 6:
        v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
        v30 = *(_QWORD *)(v9 + 88);
        break;
      case 9:
      case 0xA:
        v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
        v30 = *(_QWORD *)(v9 + 88);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v29 = *(_QWORD *)(v9 + 88);
        v30 = v29;
        break;
      default:
        v6 = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 176LL);
        v7 = sub_725FD0();
        v8 = sub_88DE40(v6, v51);
        v7[12].m128i_i8[3] |= 1u;
        v7[9].m128i_i64[1] = (__int64)v8;
        BUG();
    }
    v31 = *(_QWORD *)(v30 + 176);
    v49 = v29;
    v32 = sub_725FD0();
    v33 = sub_88DE40(v31, v51);
    v32[12].m128i_i8[3] |= 1u;
    v32[9].m128i_i64[1] = (__int64)v33;
    v32[15].m128i_i64[1] = *(_QWORD *)(v49 + 104);
    v34 = sub_8A3C00(**(_QWORD **)(v49 + 328), 0, 0, 0);
    v32[15].m128i_i64[0] = (__int64)v34;
    v35 = (__int64 *)v34;
    v55[0] = (__int64 *)v34;
    if ( v34 )
    {
      do
      {
        if ( *((_BYTE *)v35 + 8) == 3 )
        {
          sub_72F220(v55);
          v35 = v55[0];
          if ( !v55[0] )
            break;
        }
        sub_895F30((__int64)v35);
        v35 = (__int64 *)*v55[0];
        v55[0] = v35;
      }
      while ( v35 );
    }
    sub_725ED0((__int64)v32, *(_BYTE *)(v31 + 174));
    if ( v32[10].m128i_i8[14] == 5 )
      v32[11].m128i_i8[0] = *(_BYTE *)(v31 + 176);
    v36 = sub_72C930();
    v21 = sub_87F4B0(v9, &dword_4F077C8, v36);
    if ( v51 )
      sub_877E20((__int64)v21, (__int64)v32, v51, v37, v38, v39);
    *((_BYTE *)v21 + 81) |= 0x20u;
    v21[11] = v32;
    sub_877D80((__int64)v32, v21);
    v40 = sub_880C60();
    *(_BYTE *)(v40 + 80) |= 2u;
    *(_QWORD *)(v40 + 32) = v9;
    *(_QWORD *)(v40 + 24) = v21;
    v21[12] = v40;
    sub_729730(v54);
    goto LABEL_26;
  }
  v28 = sub_8B6180(v9, (__m128i *)*a2, 0);
  v21 = v28;
  if ( (*(_BYTE *)(v23 + 194) & 0x40) != 0 )
  {
    v41 = v28[11];
    v42 = **(_QWORD **)(v41 + 232);
    v55[0] = (__int64 *)sub_72F240((const __m128i *)*a2);
    if ( *(_BYTE *)(v42 + 80) == 10 )
      v42 = *(_QWORD *)(*(_QWORD *)(v42 + 96) + 32LL);
    v43 = *(_QWORD *)(sub_8B74F0(v42, v55, a3, a4) + 88);
    v44 = *(_QWORD *)(v41 + 152);
    *(_QWORD *)(v41 + 232) = v43;
    v45 = **(__int64 ****)(v44 + 168);
    v46 = **(__int64 ***)(*(_QWORD *)(v43 + 152) + 168LL);
    if ( v46 && v45 )
    {
      do
      {
        if ( ((_BYTE)v45[4] & 4) != 0 )
          v45[6] = v46;
        v45 = (__int64 **)*v45;
        v46 = (__int64 *)*v46;
      }
      while ( v45 && v46 );
    }
  }
  sub_88FC00(v21[11], *a2, a3);
LABEL_27:
  *a2 = 0;
  return v21;
}
