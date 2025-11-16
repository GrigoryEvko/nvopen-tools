// Function: sub_1BA8260
// Address: 0x1ba8260
//
unsigned __int64 __fastcall sub_1BA8260(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  unsigned __int64 v5; // rdi
  __int64 v6; // rsi
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  char v13; // r14
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  __int64 v18; // rax
  int v19; // r12d
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  _DWORD *v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rax
  _DWORD *v25; // r10
  _DWORD *v26; // r9
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdx
  char v30; // r8
  unsigned int v31; // eax
  __int64 *v32; // [rsp+10h] [rbp-130h]
  __int64 v33; // [rsp+18h] [rbp-128h]
  unsigned int v34; // [rsp+24h] [rbp-11Ch]
  __int64 v35; // [rsp+30h] [rbp-110h]
  __int64 *v36; // [rsp+38h] [rbp-108h]
  unsigned __int8 v37; // [rsp+42h] [rbp-FEh]
  char v38; // [rsp+43h] [rbp-FDh]
  unsigned int v39; // [rsp+44h] [rbp-FCh]
  unsigned __int64 v40; // [rsp+48h] [rbp-F8h]
  __m128i v41; // [rsp+50h] [rbp-F0h]
  _BYTE v42[16]; // [rsp+60h] [rbp-E0h] BYREF
  void (__fastcall *v43)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-D0h]
  unsigned __int8 (__fastcall *v44)(_BYTE *, __int64); // [rsp+78h] [rbp-C8h]
  __int64 v45; // [rsp+80h] [rbp-C0h]
  __int64 v46; // [rsp+88h] [rbp-B8h]
  _BYTE v47[16]; // [rsp+90h] [rbp-B0h] BYREF
  void (__fastcall *v48)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-A0h]
  __int64 v49; // [rsp+A8h] [rbp-98h]
  __m128i v50; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE v51[16]; // [rsp+C0h] [rbp-80h] BYREF
  void (__fastcall *v52)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-70h]
  unsigned __int8 (__fastcall *v53)(_BYTE *, __int64); // [rsp+D8h] [rbp-68h]
  __int64 v54; // [rsp+E0h] [rbp-60h]
  __int64 v55; // [rsp+E8h] [rbp-58h]
  _BYTE v56[16]; // [rsp+F0h] [rbp-50h] BYREF
  void (__fastcall *v57)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-40h]
  __int64 v58; // [rsp+108h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 296);
  v32 = *(__int64 **)(v2 + 40);
  if ( v32 == *(__int64 **)(v2 + 32) )
  {
    v37 = 0;
    v34 = 0;
    return ((unsigned __int64)v37 << 32) | v34;
  }
  v36 = *(__int64 **)(v2 + 32);
  v37 = 0;
  v34 = 0;
  v35 = a1 + 560;
  do
  {
    v5 = (unsigned __int64)&v50;
    v33 = *v36;
    sub_1580910(&v50);
    v43 = 0;
    v41 = v50;
    if ( v52 )
    {
      v5 = (unsigned __int64)v42;
      v52(v42, v51, 2);
      v44 = v53;
      v43 = v52;
    }
    v48 = 0;
    v45 = v54;
    v46 = v55;
    if ( v57 )
    {
      v5 = (unsigned __int64)v47;
      v57(v47, v56, 2);
      v6 = v41.m128i_i64[0];
      v49 = v58;
      v7 = v57;
      v48 = v57;
      if ( v41.m128i_i64[0] == v45 )
      {
        v39 = 0;
        v13 = 0;
        goto LABEL_25;
      }
    }
    else
    {
      v6 = v41.m128i_i64[0];
      if ( v45 == v41.m128i_i64[0] )
      {
        v39 = 0;
        v13 = 0;
        goto LABEL_27;
      }
    }
    v38 = 0;
    v39 = 0;
    do
    {
      v8 = *(_QWORD **)(a1 + 408);
      if ( v6 )
        v6 -= 24;
      v9 = *(_QWORD **)(a1 + 400);
      if ( v8 == v9 )
      {
        v10 = &v9[*(unsigned int *)(a1 + 420)];
        if ( v9 == v10 )
        {
          v12 = *(_QWORD *)(a1 + 400);
        }
        else
        {
          do
          {
            if ( v6 == *v9 )
              break;
            ++v9;
          }
          while ( v10 != v9 );
          v12 = (unsigned __int64)v10;
        }
      }
      else
      {
        v5 = a1 + 392;
        v10 = &v8[*(unsigned int *)(a1 + 416)];
        v9 = sub_16CC9F0(a1 + 392, v6);
        if ( v6 == *v9 )
        {
          v28 = *(_QWORD *)(a1 + 408);
          if ( v28 == *(_QWORD *)(a1 + 400) )
            v5 = *(unsigned int *)(a1 + 420);
          else
            v5 = *(unsigned int *)(a1 + 416);
          v12 = v28 + 8 * v5;
        }
        else
        {
          v11 = *(_QWORD *)(a1 + 408);
          if ( v11 != *(_QWORD *)(a1 + 400) )
          {
            v12 = *(unsigned int *)(a1 + 416);
            v9 = (_QWORD *)(v11 + 8 * v12);
            goto LABEL_14;
          }
          v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a1 + 420));
          v12 = (unsigned __int64)v9;
        }
      }
      for ( ; (_QWORD *)v12 != v9; ++v9 )
      {
        if ( *v9 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
LABEL_14:
      if ( v9 != v10 )
        goto LABEL_15;
      if ( a2 > 1 )
      {
        v15 = *(_QWORD **)(a1 + 576);
        v16 = *(_QWORD **)(a1 + 568);
        if ( v15 == v16 )
        {
          v17 = &v16[*(unsigned int *)(a1 + 588)];
          if ( v16 == v17 )
          {
            v12 = *(_QWORD *)(a1 + 568);
          }
          else
          {
            do
            {
              if ( v6 == *v16 )
                break;
              ++v16;
            }
            while ( v17 != v16 );
            v12 = (unsigned __int64)v17;
          }
        }
        else
        {
          v5 = v35;
          v17 = &v15[*(unsigned int *)(a1 + 584)];
          v16 = sub_16CC9F0(v35, v6);
          if ( v6 == *v16 )
          {
            v29 = *(_QWORD *)(a1 + 576);
            if ( v29 == *(_QWORD *)(a1 + 568) )
              v5 = *(unsigned int *)(a1 + 588);
            else
              v5 = *(unsigned int *)(a1 + 584);
            v12 = v29 + 8 * v5;
          }
          else
          {
            v18 = *(_QWORD *)(a1 + 576);
            if ( v18 != *(_QWORD *)(a1 + 568) )
            {
              v12 = *(unsigned int *)(a1 + 584);
              v16 = (_QWORD *)(v18 + 8 * v12);
              goto LABEL_41;
            }
            v16 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 588));
            v12 = (unsigned __int64)v16;
          }
        }
        while ( (_QWORD *)v12 != v16 && *v16 >= 0xFFFFFFFFFFFFFFFELL )
          ++v16;
LABEL_41:
        if ( v17 != v16 )
          goto LABEL_15;
      }
      v40 = sub_1BA7710(a1, v6, a2);
      v19 = v40;
      v20 = sub_16D5D50();
      v21 = *(_QWORD **)&dword_4FA0208[2];
      if ( *(_QWORD *)&dword_4FA0208[2] )
      {
        v22 = dword_4FA0208;
        do
        {
          while ( 1 )
          {
            v23 = v21[2];
            v12 = v21[3];
            if ( v20 <= v21[4] )
              break;
            v21 = (_QWORD *)v21[3];
            if ( !v12 )
              goto LABEL_47;
          }
          v22 = v21;
          v21 = (_QWORD *)v21[2];
        }
        while ( v23 );
LABEL_47:
        if ( v22 != dword_4FA0208 && v20 >= *((_QWORD *)v22 + 4) )
        {
          v24 = *((_QWORD *)v22 + 7);
          v25 = v22 + 12;
          if ( v24 )
          {
            v26 = v22 + 12;
            do
            {
              while ( 1 )
              {
                v27 = *(_QWORD *)(v24 + 16);
                v12 = *(_QWORD *)(v24 + 24);
                if ( *(_DWORD *)(v24 + 32) >= dword_4FB86A8 )
                  break;
                v24 = *(_QWORD *)(v24 + 24);
                if ( !v12 )
                  goto LABEL_54;
              }
              v26 = (_DWORD *)v24;
              v24 = *(_QWORD *)(v24 + 16);
            }
            while ( v27 );
LABEL_54:
            if ( v26 != v25 && dword_4FB86A8 >= v26[8] && (int)v26[9] > 0 )
              v19 = dword_4FB8740;
          }
        }
      }
      v5 = HIDWORD(v40);
      v39 += v19;
      v38 |= BYTE4(v40);
LABEL_15:
      v6 = *(_QWORD *)(v41.m128i_i64[0] + 8);
      v41.m128i_i64[0] = v6;
      if ( v6 != v41.m128i_i64[1] )
      {
        while ( 1 )
        {
          if ( v6 )
            v6 -= 24;
          if ( !v43 )
            sub_4263D6(v5, v6, v12);
          v5 = (unsigned __int64)v42;
          if ( v44(v42, v6) )
            break;
          v6 = *(_QWORD *)(v41.m128i_i64[0] + 8);
          v41.m128i_i64[0] = v6;
          if ( v41.m128i_i64[1] == v6 )
            goto LABEL_23;
        }
        v6 = v41.m128i_i64[0];
      }
LABEL_23:
      ;
    }
    while ( v45 != v6 );
    v7 = v48;
    v13 = v38;
LABEL_25:
    if ( v7 )
      v7(v47, v47, 3);
LABEL_27:
    if ( v43 )
      v43(v42, v42, 3);
    if ( v57 )
      v57(v56, v56, 3);
    if ( v52 )
      v52(v51, v51, 3);
    if ( a2 == 1 )
    {
      v30 = sub_1BF29F0(*(_QWORD *)(a1 + 320), v33);
      v31 = v39 >> 1;
      if ( !v30 )
        v31 = v39;
      v39 = v31;
    }
    ++v36;
    v37 |= v13;
    v34 += v39;
  }
  while ( v32 != v36 );
  return ((unsigned __int64)v37 << 32) | v34;
}
