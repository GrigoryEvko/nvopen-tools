// Function: sub_20293A0
// Address: 0x20293a0
//
void __fastcall sub_20293A0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  __int64 v8; // rsi
  int v9; // r8d
  __int64 v10; // r9
  __int64 v11; // rax
  char v12; // dl
  const void **v13; // rax
  unsigned int v14; // r12d
  __int64 v15; // r13
  unsigned int v16; // r14d
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // r9
  const void **v20; // rdx
  const void **v21; // rbx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // r10
  unsigned int v25; // edx
  unsigned __int8 v26; // al
  __int64 v27; // rsi
  __int128 v28; // rax
  __int64 *v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 **v32; // rax
  __int64 v33; // r13
  __int64 *v34; // rdi
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // r12
  __int64 v40; // r13
  __int64 *v41; // rdi
  __int64 v42; // rbx
  _QWORD *v43; // rax
  __int64 v44; // rdx
  int v45; // edx
  __int64 v46; // rax
  _QWORD *v47; // rbx
  int v48; // edx
  int v49; // r12d
  __int64 *v50; // rax
  _BYTE *v51; // rdi
  int v52; // edx
  __int128 v53; // [rsp-10h] [rbp-2E0h]
  __int64 *v55; // [rsp+30h] [rbp-2A0h]
  __int64 (__fastcall *v56)(__int64, __int64); // [rsp+38h] [rbp-298h]
  __int64 *v57; // [rsp+40h] [rbp-290h]
  __int64 *v59; // [rsp+50h] [rbp-280h]
  int v60; // [rsp+50h] [rbp-280h]
  __int64 *v61; // [rsp+50h] [rbp-280h]
  __int64 *v62; // [rsp+50h] [rbp-280h]
  __int64 v63; // [rsp+58h] [rbp-278h]
  __int64 v64; // [rsp+58h] [rbp-278h]
  unsigned int v66; // [rsp+80h] [rbp-250h]
  unsigned int v67; // [rsp+80h] [rbp-250h]
  _QWORD *v68; // [rsp+80h] [rbp-250h]
  __int64 v69; // [rsp+88h] [rbp-248h]
  int v70; // [rsp+90h] [rbp-240h]
  unsigned int v71; // [rsp+94h] [rbp-23Ch]
  unsigned int v72; // [rsp+98h] [rbp-238h]
  unsigned int v73; // [rsp+9Ch] [rbp-234h]
  __int64 v74; // [rsp+D0h] [rbp-200h] BYREF
  int v75; // [rsp+D8h] [rbp-1F8h]
  __int64 v76; // [rsp+E0h] [rbp-1F0h] BYREF
  const void **v77; // [rsp+E8h] [rbp-1E8h]
  __int64 v78; // [rsp+F0h] [rbp-1E0h] BYREF
  int v79; // [rsp+F8h] [rbp-1D8h]
  __int128 v80; // [rsp+100h] [rbp-1D0h] BYREF
  __int128 v81; // [rsp+110h] [rbp-1C0h] BYREF
  __int128 v82; // [rsp+120h] [rbp-1B0h] BYREF
  __int128 v83; // [rsp+130h] [rbp-1A0h] BYREF
  _DWORD *v84; // [rsp+140h] [rbp-190h] BYREF
  __int64 v85; // [rsp+148h] [rbp-188h]
  _BYTE v86[64]; // [rsp+150h] [rbp-180h] BYREF
  _BYTE *v87; // [rsp+190h] [rbp-140h] BYREF
  __int64 v88; // [rsp+198h] [rbp-138h]
  _BYTE v89[304]; // [rsp+1A0h] [rbp-130h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v74 = v8;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  if ( v8 )
    sub_1623A60((__int64)&v74, v8, 2);
  v75 = *(_DWORD *)(a2 + 64);
  sub_2017DE0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v80, &v81);
  sub_2017DE0(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
    &v82,
    &v83);
  v11 = *(_QWORD *)(v80 + 40) + 16LL * DWORD2(v80);
  v12 = *(_BYTE *)v11;
  v13 = *(const void ***)(v11 + 8);
  LOBYTE(v76) = v12;
  v77 = v13;
  if ( v12 )
    v71 = word_4305480[(unsigned __int8)(v12 - 14)];
  else
    v71 = sub_1F58D30((__int64)&v76);
  v70 = 2;
  v84 = v86;
  v85 = 0x1000000000LL;
  v73 = v71;
  while ( 1 )
  {
    v14 = v73 - v71;
    v72 = v73 - v71;
    if ( !v71 )
      goto LABEL_52;
    v15 = 0xFFFFFFFFLL;
    v16 = -1;
    do
    {
      v17 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 4LL * v14) / v71;
      if ( v17 > 3 )
      {
        v46 = (unsigned int)v85;
        if ( (unsigned int)v85 >= HIDWORD(v85) )
        {
          sub_16CD150((__int64)&v84, v86, 0, 4, v9, v10);
          v46 = (unsigned int)v85;
        }
        v84[v46] = -1;
        LODWORD(v85) = v85 + 1;
      }
      else
      {
        v10 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 4LL * v14) % v71;
        if ( (_DWORD)v15 != v17 )
        {
          if ( (_DWORD)v15 == -1 )
          {
            v15 = v17;
          }
          else if ( v17 == v16 )
          {
            v10 = v71 + (unsigned int)v10;
          }
          else
          {
            if ( v16 != -1 )
            {
              LOBYTE(v18) = sub_1F7E0F0((__int64)&v76);
              v87 = v89;
              v66 = v18;
              v21 = v20;
              v88 = 0x1000000000LL;
              while ( 1 )
              {
                v33 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 4LL * v72) / v71;
                v60 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 4LL * v72) % v71;
                if ( (unsigned int)v33 <= 3 )
                {
                  v57 = a1[1];
                  v55 = *a1;
                  v56 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
                  v22 = sub_1E0A0C0(v57[4]);
                  if ( v56 == sub_1D13A20 )
                  {
                    v23 = sub_15A9520(v22, 0);
                    v24 = (__int64)v57;
                    v25 = 8 * v23;
                    if ( 8 * v23 == 32 )
                    {
                      v26 = 5;
                    }
                    else if ( v25 > 0x20 )
                    {
                      v26 = 6;
                      if ( v25 != 64 )
                      {
                        v26 = 0;
                        if ( v25 == 128 )
                          v26 = 7;
                      }
                    }
                    else
                    {
                      v26 = 3;
                      if ( v25 != 8 )
                        v26 = 4 * (v25 == 16);
                    }
                  }
                  else
                  {
                    v26 = v56((__int64)v55, v22);
                    v24 = (__int64)v57;
                  }
                  v27 = v60;
                  v59 = (__int64 *)v24;
                  *(_QWORD *)&v28 = sub_1D38BB0(v24, v27, (__int64)&v74, v26, 0, 0, (__m128i)0LL, a6, a7, 0);
                  v29 = sub_1D332F0(
                          v59,
                          106,
                          (__int64)&v74,
                          v66,
                          v21,
                          0,
                          0.0,
                          a6,
                          a7,
                          *((_QWORD *)&v80 + 2 * v33),
                          *((_QWORD *)&v80 + 2 * v33 + 1),
                          v28);
                  v19 = v30;
                  v31 = (unsigned int)v88;
                  if ( (unsigned int)v88 < HIDWORD(v88) )
                    goto LABEL_20;
                }
                else
                {
                  v34 = a1[1];
                  v78 = 0;
                  v79 = 0;
                  v35 = sub_1D2B300(v34, 0x30u, (__int64)&v78, v66, (__int64)v21, v19);
                  v29 = v35;
                  v19 = v36;
                  if ( v78 )
                  {
                    v61 = v35;
                    v63 = v36;
                    sub_161E7C0((__int64)&v78, v78);
                    v29 = v61;
                    v19 = v63;
                  }
                  v31 = (unsigned int)v88;
                  if ( (unsigned int)v88 < HIDWORD(v88) )
                    goto LABEL_20;
                }
                v62 = v29;
                v64 = v19;
                sub_16CD150((__int64)&v87, v89, 0, 16, (int)v29, v19);
                v31 = (unsigned int)v88;
                v29 = v62;
                v19 = v64;
LABEL_20:
                v32 = (__int64 **)&v87[16 * v31];
                *v32 = v29;
                v32[1] = (__int64 *)v19;
                ++v72;
                LODWORD(v88) = v88 + 1;
                if ( v73 == v72 )
                {
                  *((_QWORD *)&v53 + 1) = (unsigned int)v88;
                  *(_QWORD *)&v53 = v87;
                  v50 = sub_1D359D0(a1[1], 104, (__int64)&v74, v76, v77, 0, 0.0, a6, a7, v53);
                  v51 = v87;
                  *(_QWORD *)a3 = v50;
                  *(_DWORD *)(a3 + 8) = v52;
                  if ( v51 != v89 )
                    _libc_free((unsigned __int64)v51);
                  goto LABEL_35;
                }
              }
            }
            v10 = v71 + (unsigned int)v10;
            v16 = *(_DWORD *)(*(_QWORD *)(a2 + 88) + 4LL * v14) / v71;
          }
        }
        v37 = (unsigned int)v85;
        if ( (unsigned int)v85 >= HIDWORD(v85) )
        {
          v67 = v10;
          sub_16CD150((__int64)&v84, v86, 0, 4, v9, v10);
          v37 = (unsigned int)v85;
          v10 = v67;
        }
        v84[v37] = v10;
        LODWORD(v85) = v85 + 1;
      }
      ++v14;
    }
    while ( v73 != v14 );
    if ( (_DWORD)v15 != -1 )
    {
      v38 = 16 * v15;
      v39 = *(_QWORD *)((char *)&v80 + v38);
      v40 = *(_QWORD *)((char *)&v80 + v38 + 8);
      v41 = a1[1];
      if ( v16 == -1 )
      {
        v87 = 0;
        LODWORD(v88) = 0;
        v43 = sub_1D2B300(v41, 0x30u, (__int64)&v87, v76, (__int64)v77, v10);
        if ( v87 )
        {
          v68 = v43;
          v69 = v44;
          sub_161E7C0((__int64)&v87, (__int64)v87);
          v43 = v68;
          v44 = v69;
        }
        v41 = a1[1];
      }
      else
      {
        v42 = 16LL * v16;
        v43 = *(_QWORD **)((char *)&v80 + v42);
        v44 = *(_QWORD *)((char *)&v80 + v42 + 8);
      }
      *(_QWORD *)a3 = sub_1D41320(
                        (__int64)v41,
                        (unsigned int)v76,
                        v77,
                        (__int64)&v74,
                        v39,
                        v40,
                        0.0,
                        a6,
                        a7,
                        (__int64)v43,
                        v44,
                        v84,
                        (unsigned int)v85);
      *(_DWORD *)(a3 + 8) = v45;
      goto LABEL_35;
    }
LABEL_52:
    v87 = 0;
    LODWORD(v88) = 0;
    v47 = sub_1D2B300(a1[1], 0x30u, (__int64)&v87, v76, (__int64)v77, v10);
    v49 = v48;
    if ( v87 )
      sub_161E7C0((__int64)&v87, (__int64)v87);
    *(_QWORD *)a3 = v47;
    *(_DWORD *)(a3 + 8) = v49;
LABEL_35:
    LODWORD(v85) = 0;
    v73 += v71;
    if ( v70 == 1 )
      break;
    v70 = 1;
    a3 = a4;
  }
  if ( v84 != (_DWORD *)v86 )
    _libc_free((unsigned __int64)v84);
  if ( v74 )
    sub_161E7C0((__int64)&v74, v74);
}
