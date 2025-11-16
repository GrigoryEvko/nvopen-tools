// Function: sub_1A96B10
// Address: 0x1a96b10
//
void __fastcall sub_1A96B10(
        __int64 a1,
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
  _QWORD *v10; // r14
  __int64 *v11; // r12
  _DWORD *v12; // rbx
  _DWORD *v13; // r15
  int v14; // eax
  _QWORD *v15; // r13
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int64 v18; // r8
  int v19; // r9d
  _QWORD *v20; // rbx
  _QWORD *v21; // r14
  _BYTE *i; // r13
  _QWORD *v23; // r14
  unsigned __int8 v24; // al
  __int64 ***v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 ****v29; // rbx
  __int64 ****v30; // r12
  __int64 ***v31; // r14
  __int64 v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  unsigned __int64 v35; // r15
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // r15
  int v41; // ebx
  __int64 v42; // r13
  int v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rax
  _QWORD *v46; // [rsp+8h] [rbp-108h]
  _QWORD *v47; // [rsp+10h] [rbp-100h]
  _QWORD *v48; // [rsp+20h] [rbp-F0h]
  __int64 *v49; // [rsp+30h] [rbp-E0h]
  _QWORD *v50; // [rsp+38h] [rbp-D8h]
  __int64 v51; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v52[3]; // [rsp+50h] [rbp-C0h] BYREF
  int v53; // [rsp+68h] [rbp-A8h]
  __int64 ****v54; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v55; // [rsp+78h] [rbp-98h]
  _BYTE v56[144]; // [rsp+80h] [rbp-90h] BYREF

  v10 = *(_QWORD **)(a1 + 32);
  if ( v10 != (_QWORD *)(a1 + 24) )
  {
    do
    {
      if ( !v10 )
      {
        sub_15E0530(0);
        BUG();
      }
      v11 = (__int64 *)sub_15E0530((__int64)(v10 - 7));
      if ( (*((_BYTE *)v10 - 38) & 1) != 0 )
      {
        sub_15E08E0((__int64)(v10 - 7), a2);
        v12 = (_DWORD *)v10[4];
        v13 = &v12[10 * v10[5]];
        if ( (*((_BYTE *)v10 - 38) & 1) != 0 )
        {
          sub_15E08E0((__int64)(v10 - 7), a2);
          v12 = (_DWORD *)v10[4];
        }
      }
      else
      {
        v12 = (_DWORD *)v10[4];
        v13 = &v12[10 * v10[5]];
      }
      while ( v13 != v12 )
      {
        while ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 15 )
        {
          v12 += 10;
          if ( v13 == v12 )
            goto LABEL_10;
        }
        v14 = v12[8];
        a2 = (__int64)(v10 - 7);
        v12 += 10;
        sub_1A95C60(v11, (__int64)(v10 - 7), v14 + 1);
      }
LABEL_10:
      if ( *(_BYTE *)(**(_QWORD **)(*(v10 - 4) + 16LL) + 8LL) == 15 )
      {
        a2 = (__int64)(v10 - 7);
        sub_1A95C60(v11, (__int64)(v10 - 7), 0);
      }
      v10 = (_QWORD *)v10[1];
    }
    while ( (_QWORD *)(a1 + 24) != v10 );
    v15 = *(_QWORD **)(a1 + 32);
    if ( v10 != v15 )
    {
      v48 = v10;
      do
      {
        if ( !v15 )
          BUG();
        v16 = v15 + 2;
        if ( v15 + 2 != (_QWORD *)(v15[2] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v17 = sub_15E0530((__int64)(v15 - 7));
          v20 = (_QWORD *)v15[3];
          v49 = (__int64 *)v17;
          v51 = v17;
          v54 = (__int64 ****)v56;
          v55 = 0xC00000000LL;
          if ( v16 == v20 )
          {
            v21 = 0;
          }
          else
          {
            if ( !v20 )
              BUG();
            while ( 1 )
            {
              v21 = (_QWORD *)v20[3];
              if ( v21 != v20 + 2 )
                break;
              v20 = (_QWORD *)v20[1];
              if ( v16 == v20 )
                break;
              if ( !v20 )
                BUG();
            }
          }
          v50 = v15;
          i = v21;
          v23 = v16;
LABEL_24:
          while ( v23 != v20 )
          {
            if ( !i )
              BUG();
            v24 = *(i - 8);
            v25 = (__int64 ***)(i - 24);
            if ( v24 == 78
              && (v44 = *((_QWORD *)i - 6), !*(_BYTE *)(v44 + 16))
              && (*(_BYTE *)(v44 + 33) & 0x20) != 0
              && *(_DWORD *)(v44 + 36) == 114 )
            {
              v45 = (unsigned int)v55;
              if ( (unsigned int)v55 >= HIDWORD(v55) )
              {
                sub_16CD150((__int64)&v54, v56, 0, 8, v18, v19);
                v45 = (unsigned int)v55;
              }
              v54[v45] = v25;
              LODWORD(v55) = v55 + 1;
            }
            else
            {
              if ( *((_QWORD *)i + 3) || *((__int16 *)i - 3) < 0 )
              {
                v26 = sub_1625790((__int64)(i - 24), 1);
                if ( v26 )
                {
                  v27 = sub_161C790(&v51, v26);
                  sub_1625C10((__int64)(i - 24), 1, v27);
                }
                v24 = *(i - 8);
              }
              if ( (unsigned __int8)(v24 - 54) <= 1u )
              {
                v52[0] = 0x400000001LL;
                v52[1] = 0x900000007LL;
                v52[2] = 0x110000000BLL;
                v53 = 19;
                sub_1624960((__int64)(i - 24), (unsigned int *)v52, 7);
                v24 = *(i - 8);
              }
              if ( v24 > 0x17u )
              {
                if ( v24 == 78 )
                {
                  v35 = (unsigned __int64)v25 & 0xFFFFFFFFFFFFFFF8LL;
                  v36 = (unsigned __int64)(i - 24) | 4;
                  goto LABEL_54;
                }
                if ( v24 == 29 )
                {
                  v35 = (unsigned __int64)v25 & 0xFFFFFFFFFFFFFFF8LL;
                  v36 = (unsigned __int64)(i - 24) & 0xFFFFFFFFFFFFFFFBLL;
LABEL_54:
                  v52[0] = v36;
                  if ( v35 )
                  {
                    v37 = sub_1389B50(v52);
                    v18 = v52[0];
                    v38 = v52[0] & 0xFFFFFFFFFFFFFFF8LL;
                    v39 = 0xAAAAAAAAAAAAAAABLL
                        * ((__int64)(v37
                                   - ((v52[0] & 0xFFFFFFFFFFFFFFF8LL)
                                    - 24LL * (*(_DWORD *)((v52[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
                    if ( (_DWORD)v39 )
                    {
                      v40 = 0;
                      v47 = v20;
                      v41 = 1;
                      v46 = i;
                      v42 = 8 * (3LL * (unsigned int)(v39 - 1) + 3);
                      do
                      {
                        while ( 1 )
                        {
                          v38 = v18 & 0xFFFFFFFFFFFFFFF8LL;
                          if ( *(_BYTE *)(**(_QWORD **)((v18 & 0xFFFFFFFFFFFFFFF8LL)
                                                      + v40
                                                      - 24LL
                                                      * (*(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))
                                        + 8LL) == 15 )
                            break;
                          v40 += 24;
                          ++v41;
                          if ( v42 == v40 )
                            goto LABEL_60;
                        }
                        v43 = v41;
                        v40 += 24;
                        ++v41;
                        sub_1A95AA0(v49, v52, v43);
                        v18 = v52[0];
                        v38 = v52[0] & 0xFFFFFFFFFFFFFFF8LL;
                      }
                      while ( v42 != v40 );
LABEL_60:
                      v20 = v47;
                      i = v46;
                    }
                    if ( *(_BYTE *)(*(_QWORD *)v38 + 8LL) == 15 )
                      sub_1A95AA0(v49, v52, 0);
                  }
                }
              }
            }
            for ( i = (_BYTE *)*((_QWORD *)i + 1); ; i = (_BYTE *)v20[3] )
            {
              v28 = v20 - 3;
              if ( !v20 )
                v28 = 0;
              if ( i != (_BYTE *)(v28 + 5) )
                break;
              v20 = (_QWORD *)v20[1];
              if ( v23 == v20 )
                goto LABEL_24;
              if ( !v20 )
                BUG();
            }
          }
          v29 = v54;
          v15 = v50;
          v30 = &v54[(unsigned int)v55];
          if ( v54 != v30 )
          {
            do
            {
              v31 = *v29++;
              v32 = sub_1599EF0(*v31);
              sub_164D160((__int64)v31, v32, a3, a4, a5, a6, v33, v34, a9, a10);
              sub_15F20C0(v31);
            }
            while ( v30 != v29 );
            v30 = v54;
          }
          if ( v30 != (__int64 ****)v56 )
            _libc_free((unsigned __int64)v30);
        }
        v15 = (_QWORD *)v15[1];
      }
      while ( v48 != v15 );
    }
  }
}
