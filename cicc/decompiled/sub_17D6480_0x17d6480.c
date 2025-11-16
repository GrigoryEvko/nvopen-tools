// Function: sub_17D6480
// Address: 0x17d6480
//
_QWORD *__fastcall sub_17D6480(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // r13
  __int64 *v12; // rax
  __int64 **v13; // rdx
  __int128 v14; // rdi
  __int64 *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r15
  __int64 v19; // r13
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 *v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v46; // [rsp+10h] [rbp-110h]
  unsigned __int64 v47; // [rsp+30h] [rbp-F0h]
  int v48; // [rsp+3Ch] [rbp-E4h]
  __int64 v49; // [rsp+40h] [rbp-E0h]
  signed int v50; // [rsp+50h] [rbp-D0h]
  __int64 **v51; // [rsp+58h] [rbp-C8h]
  __int64 v52; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v53[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int16 v54; // [rsp+80h] [rbp-A0h]
  _QWORD v55[2]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v56; // [rsp+A0h] [rbp-80h]
  __int64 *v57; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v58; // [rsp+C0h] [rbp-60h] BYREF
  int v59; // [rsp+D0h] [rbp-50h]

  v49 = sub_1632FA0(*(_QWORD *)(a1[1] + 40LL));
  v5 = (*a2 & 0xFFFFFFFFFFFFFFF8LL)
     - 24LL * (*(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)
     + 24LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 64) + 12LL) - 1);
  v51 = (__int64 **)v5;
  v47 = sub_1389B50(a2);
  if ( v5 == v47 )
  {
    v42 = 0;
  }
  else
  {
    v50 = 0;
    do
    {
      v17 = *(_QWORD *)(a1[1] + 40LL);
      v56 = 260;
      v55[0] = v17 + 240;
      sub_16E1010((__int64)&v57, (__int64)v55);
      v18 = *v51;
      v19 = **v51;
      v20 = (unsigned int)sub_15A9FE0(v49, v19);
      v21 = v20 * ((v20 + ((unsigned __int64)(sub_127FA20(v49, v19) + 7) >> 3) - 1) / v20);
      if ( v59 == 12 && v21 <= 7 )
      {
        v48 = v50 + 8;
        v50 = v50 + 8 - v21;
      }
      else
      {
        v48 = v50 + v21;
      }
      v22 = a1[2];
      v23 = *v18;
      v54 = 257;
      v6 = *(_QWORD *)(v22 + 224);
      v24 = *(_QWORD *)(v22 + 176);
      v7 = *(_QWORD *)v6;
      if ( v24 != *(_QWORD *)v6 )
      {
        if ( *(_BYTE *)(v6 + 16) <= 0x10u )
        {
          v6 = sub_15A4A70(*(__int64 ****)(v22 + 224), v24);
          v7 = *(_QWORD *)(a1[2] + 176LL);
        }
        else
        {
          v25 = *(_QWORD *)(v22 + 224);
          v56 = 257;
          v26 = sub_15FDFF0(v25, v24, (__int64)v55, 0);
          v27 = *(_QWORD *)(a3 + 8);
          v6 = v26;
          if ( v27 )
          {
            v46 = *(__int64 **)(a3 + 16);
            sub_157E9D0(v27 + 40, v26);
            v28 = *v46;
            v29 = *(_QWORD *)(v6 + 24) & 7LL;
            *(_QWORD *)(v6 + 32) = v46;
            v28 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v6 + 24) = v28 | v29;
            *(_QWORD *)(v28 + 8) = v6 + 24;
            *v46 = *v46 & 7 | (v6 + 24);
          }
          sub_164B780(v6, v53);
          v30 = *(_QWORD *)a3;
          if ( *(_QWORD *)a3 )
          {
            v52 = *(_QWORD *)a3;
            sub_1623A60((__int64)&v52, v30, 2);
            v31 = *(_QWORD *)(v6 + 48);
            v32 = v6 + 48;
            if ( v31 )
            {
              sub_161E7C0(v6 + 48, v31);
              v32 = v6 + 48;
            }
            v33 = (unsigned __int8 *)v52;
            *(_QWORD *)(v6 + 48) = v52;
            if ( v33 )
              sub_1623210((__int64)&v52, v33, v32);
          }
          v7 = *(_QWORD *)(a1[2] + 176LL);
        }
      }
      v56 = 257;
      v8 = sub_15A0680(v7, v50, 0);
      v9 = sub_12899C0((__int64 *)a3, v6, v8, (__int64)v55, 0, 0);
      v10 = (_QWORD *)a1[3];
      v11 = v9;
      v54 = 259;
      v53[0] = (__int64)"_msarg";
      v12 = sub_17CD8D0(v10, v23);
      v13 = (__int64 **)sub_1646BA0(v12, 0);
      if ( v13 != *(__int64 ***)v11 )
      {
        if ( *(_BYTE *)(v11 + 16) > 0x10u )
        {
          v56 = 257;
          v34 = sub_15FDBD0(46, v11, (__int64)v13, (__int64)v55, 0);
          v35 = *(_QWORD *)(a3 + 8);
          v11 = v34;
          if ( v35 )
          {
            v36 = *(__int64 **)(a3 + 16);
            sub_157E9D0(v35 + 40, v34);
            v37 = *(_QWORD *)(v11 + 24);
            v38 = *v36;
            *(_QWORD *)(v11 + 32) = v36;
            v38 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v11 + 24) = v38 | v37 & 7;
            *(_QWORD *)(v38 + 8) = v11 + 24;
            *v36 = *v36 & 7 | (v11 + 24);
          }
          sub_164B780(v11, v53);
          v39 = *(_QWORD *)a3;
          if ( *(_QWORD *)a3 )
          {
            v52 = *(_QWORD *)a3;
            sub_1623A60((__int64)&v52, v39, 2);
            v40 = *(_QWORD *)(v11 + 48);
            if ( v40 )
              sub_161E7C0(v11 + 48, v40);
            v41 = (unsigned __int8 *)v52;
            *(_QWORD *)(v11 + 48) = v52;
            if ( v41 )
              sub_1623210((__int64)&v52, v41, v11 + 48);
          }
        }
        else
        {
          v11 = sub_15A46C0(46, (__int64 ***)v11, v13, 0);
        }
      }
      *(_QWORD *)&v14 = a1[3];
      *((_QWORD *)&v14 + 1) = v18;
      v50 = (v48 + 7) & 0xFFFFFFF8;
      v15 = sub_17D4DA0(v14);
      v16 = sub_12A8F50((__int64 *)a3, (__int64)v15, v11, 0);
      sub_15F9450((__int64)v16, 8u);
      if ( v57 != &v58 )
        j_j___libc_free_0(v57, v58 + 1);
      v51 += 3;
    }
    while ( (__int64 **)v47 != v51 );
    v42 = (v48 + 7) & 0xFFFFFFF8;
  }
  v43 = sub_1643360(*(_QWORD **)(a3 + 24));
  v44 = sub_159C470(v43, v42, 0);
  return sub_12A8F50((__int64 *)a3, v44, *(_QWORD *)(a1[2] + 232LL), 0);
}
