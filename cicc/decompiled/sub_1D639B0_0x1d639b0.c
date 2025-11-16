// Function: sub_1D639B0
// Address: 0x1d639b0
//
__int64 __fastcall sub_1D639B0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // r14
  char v7; // al
  __int64 v8; // rcx
  unsigned int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // r12d
  unsigned int v13; // r12d
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // r8
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 (*v22)(); // rax
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // r15
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  unsigned __int8 v32; // bl
  __int64 v33; // r15
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 i; // r12
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 v42; // r12
  char v43; // dl
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rsi
  _QWORD *v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rdi
  __int64 (*v51)(); // rdx
  unsigned __int64 v52; // r12
  int v53; // r8d
  int v54; // r9d
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-158h]
  unsigned __int8 v58; // [rsp+1Fh] [rbp-141h]
  __int64 v59; // [rsp+20h] [rbp-140h]
  __int64 v60; // [rsp+20h] [rbp-140h]
  __int64 v61; // [rsp+28h] [rbp-138h]
  __int64 v62; // [rsp+28h] [rbp-138h]
  __int64 v64; // [rsp+38h] [rbp-128h]
  __int64 v65; // [rsp+38h] [rbp-128h]
  _BYTE *v66; // [rsp+40h] [rbp-120h] BYREF
  __int64 v67; // [rsp+48h] [rbp-118h]
  _BYTE v68[32]; // [rsp+50h] [rbp-110h] BYREF
  __m128i v69; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD *v70; // [rsp+88h] [rbp-D8h]
  __m128i v71; // [rsp+D0h] [rbp-90h] BYREF
  _BYTE *v72; // [rsp+E0h] [rbp-80h]
  __int64 v73; // [rsp+E8h] [rbp-78h]
  int v74; // [rsp+F0h] [rbp-70h]
  _BYTE v75[104]; // [rsp+F8h] [rbp-68h] BYREF

  if ( !*(_QWORD *)(a1 + 176) )
    return 0;
  v2 = a2;
  v3 = sub_157EBA0((__int64)a2);
  v4 = v3;
  if ( *(_BYTE *)(v3 + 16) != 25 )
    return 0;
  v5 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
  if ( v5 && (v6 = *(_QWORD *)(v4 - 24LL * v5)) != 0 )
  {
    v7 = *(_BYTE *)(v6 + 16);
    v8 = 0;
    if ( v7 == 71 )
    {
      v8 = v6;
      v7 = *(_BYTE *)(*(_QWORD *)(v6 - 24) + 16LL);
      v6 = *(_QWORD *)(v6 - 24);
    }
    v9 = 0;
    if ( v7 == 77 && a2 == *(_QWORD **)(v6 + 40) )
    {
      v10 = a2[6];
      do
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_86;
        if ( *(_BYTE *)(v10 - 8) != 78 )
          break;
        v27 = *(_QWORD *)(v10 - 48);
        if ( *(_BYTE *)(v27 + 16) )
          break;
      }
      while ( (*(_BYTE *)(v27 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v27 + 36) - 35) <= 3 );
      v11 = v10 - 24;
      if ( v8 == v10 - 24 )
      {
        v56 = *(_QWORD *)(v10 + 8);
        if ( !v56 )
          return 0;
        v11 = v56 - 24;
      }
      if ( v4 == v11 )
      {
        v12 = *(_DWORD *)(v6 + 20);
        v61 = a2[7];
        v66 = v68;
        v67 = 0x400000000LL;
        v13 = v12 & 0xFFFFFFF;
        if ( v13 )
        {
          v14 = 0;
          v15 = v6;
          v16 = 8LL * v13;
          do
          {
            v17 = sub_13CF970(v15);
            v18 = *(_QWORD *)(v17 + 3 * v14);
            if ( !v18 )
              BUG();
            if ( *(_BYTE *)(v18 + 16) == 78 )
            {
              v19 = *(_QWORD *)(v18 + 8);
              if ( v19 )
              {
                if ( !*(_QWORD *)(v19 + 8) )
                {
                  v20 = (*(_BYTE *)(v15 + 23) & 0x40) != 0
                      ? *(_QWORD *)(v15 - 8)
                      : v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
                  if ( *(_QWORD *)(v14 + v20 + 24LL * *(unsigned int *)(v15 + 56) + 8) == *(_QWORD *)(v18 + 40) )
                  {
                    v21 = *(_QWORD *)(a1 + 176);
                    v22 = *(__int64 (**)())(*(_QWORD *)v21 + 1240LL);
                    if ( v22 != sub_1D5A430
                      && ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v22)(v21, *(_QWORD *)(v17 + 3 * v14))
                      && (unsigned __int8)sub_20C83A0(v61, v18, v4, *(_QWORD *)(a1 + 176), 0) )
                    {
                      v25 = (unsigned int)v67;
                      if ( (unsigned int)v67 >= HIDWORD(v67) )
                      {
                        sub_16CD150((__int64)&v66, v68, 0, 8, v23, v24);
                        v25 = (unsigned int)v67;
                      }
                      *(_QWORD *)&v66[8 * v25] = v18;
                      LODWORD(v67) = v67 + 1;
                    }
                  }
                }
              }
            }
            v14 += 8;
          }
          while ( v14 != v16 );
          v2 = a2;
LABEL_38:
          v9 = v67;
          if ( (_DWORD)v67 )
          {
            v57 = v4;
            v62 = 8LL * (unsigned int)v67;
            v58 = 0;
            v59 = (__int64)v2;
            v28 = 0;
            do
            {
              v29 = *(_QWORD *)&v66[v28];
              v64 = *(_QWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 56);
              sub_1562F70(&v71, v64, 0);
              v30 = sub_1560700(&v71, 20);
              sub_1562F70(&v69, v64, 0);
              v31 = sub_1560700(&v69, 20);
              v32 = sub_1561B70(v31, v30);
              sub_1D5AA50(v70);
              sub_1D5AA50((_QWORD *)v73);
              if ( v32 )
              {
                v33 = *(_QWORD *)(v29 + 40);
                v34 = sub_157EBA0(v33);
                if ( *(_BYTE *)(v34 + 16) == 26 && (*(_DWORD *)(v34 + 20) & 0xFFFFFFF) == 1 )
                {
                  v35 = *(_QWORD *)(v34 - 24);
                  if ( v59 == v35 )
                  {
                    if ( v35 )
                    {
                      sub_1AA6640(v57, v59, v33);
                      v58 = v32;
                      *(_BYTE *)(a1 + 896) = 1;
                    }
                  }
                }
              }
              v28 += 8;
            }
            while ( v28 != v62 );
            v9 = v58;
            if ( v58 )
            {
              if ( !*(_WORD *)(v59 + 18) )
              {
                v71.m128i_i64[0] = *(_QWORD *)(v59 + 8);
                sub_15CDD40(v71.m128i_i64);
                if ( !v71.m128i_i64[0] )
                  sub_157F980(v59);
              }
            }
          }
          if ( v66 != v68 )
            _libc_free((unsigned __int64)v66);
          return v9;
        }
      }
      return 0;
    }
  }
  else
  {
    for ( i = a2[6]; ; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
LABEL_86:
        BUG();
      if ( *(_BYTE *)(i - 8) != 78 )
        break;
      v55 = *(_QWORD *)(i - 48);
      if ( *(_BYTE *)(v55 + 16) || (*(_BYTE *)(v55 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v55 + 36) - 35) > 3 )
        break;
    }
    v37 = i - 24;
    v9 = 0;
    if ( v4 == v37 )
    {
      v38 = a2[7];
      v73 = 4;
      v71.m128i_i64[0] = 0;
      v65 = v38;
      v66 = v68;
      v67 = 0x400000000LL;
      v71.m128i_i64[1] = (__int64)v75;
      v72 = v75;
      v39 = a2[1];
      v74 = 0;
      v69.m128i_i64[0] = v39;
      sub_15CDD40(v69.m128i_i64);
      v40 = v69.m128i_i64[0];
      if ( v69.m128i_i64[0] )
      {
        v60 = v37;
        do
        {
          v41 = sub_1648700(v40);
          sub_1412190((__int64)&v71, v41[5]);
          v42 = v69.m128i_i64[0];
          if ( v43 )
          {
            v44 = sub_1648700(v69.m128i_i64[0])[5];
            v45 = *(_QWORD *)(v44 + 40);
            v46 = v44 + 40;
            v47 = (_QWORD *)(v45 & 0xFFFFFFFFFFFFFFF8LL);
            while ( 1 )
            {
              v48 = *v47 & 0xFFFFFFFFFFFFFFF8LL;
              v47 = (_QWORD *)v48;
              if ( v46 == v48 )
                break;
              if ( !v48 )
                goto LABEL_86;
              if ( *(_BYTE *)(v48 - 8) != 78 )
                break;
              v49 = *(_QWORD *)(v48 - 48);
              if ( *(_BYTE *)(v49 + 16)
                || (*(_BYTE *)(v49 + 33) & 0x20) == 0
                || (unsigned int)(*(_DWORD *)(v49 + 36) - 35) > 3 )
              {
                if ( !*(_QWORD *)(v48 - 16) )
                {
                  v50 = *(_QWORD *)(a1 + 176);
                  v51 = *(__int64 (**)())(*(_QWORD *)v50 + 1240LL);
                  if ( v51 != sub_1D5A430 )
                  {
                    v52 = v48 - 24;
                    if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v51)(v50, v48 - 24)
                      && (unsigned __int8)sub_20C83A0(v65, v52, v60, *(_QWORD *)(a1 + 176), 0) )
                    {
                      if ( (unsigned int)v67 >= HIDWORD(v67) )
                        sub_16CD150((__int64)&v66, v68, 0, 8, v53, v54);
                      *(_QWORD *)&v66[8 * (unsigned int)v67] = v52;
                      LODWORD(v67) = v67 + 1;
                    }
                    v42 = v69.m128i_i64[0];
                  }
                }
                break;
              }
            }
          }
          v69.m128i_i64[0] = *(_QWORD *)(v42 + 8);
          sub_15CDD40(v69.m128i_i64);
          v40 = v69.m128i_i64[0];
        }
        while ( v69.m128i_i64[0] );
      }
      if ( v72 != (_BYTE *)v71.m128i_i64[1] )
        _libc_free((unsigned __int64)v72);
      goto LABEL_38;
    }
  }
  return v9;
}
