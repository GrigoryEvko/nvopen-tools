// Function: sub_2FD88F0
// Address: 0x2fd88f0
//
void __fastcall sub_2FD88F0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r14
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rcx
  int v16; // eax
  int v17; // edx
  int v18; // edi
  unsigned int v19; // eax
  int v20; // esi
  __int64 v21; // r14
  int v22; // r15d
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // edx
  _DWORD *v26; // r11
  int v27; // esi
  __int64 v28; // r9
  __int64 v29; // rdi
  __int64 v30; // rcx
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v35; // r13
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rdx
  int v41; // eax
  int v42; // edx
  __int32 v43; // eax
  unsigned __int8 *v44; // rsi
  _DWORD *v45; // r11
  __int64 v46; // rax
  _QWORD *v47; // r10
  __int64 v48; // rsi
  _QWORD *v49; // rax
  __int64 v50; // r10
  _DWORD *v51; // r11
  __int64 v52; // r9
  __int64 v53; // r9
  __int64 v54; // r10
  _DWORD *v55; // r11
  __int64 v56; // r9
  __int64 v57; // r10
  _DWORD *v58; // r11
  int v59; // eax
  _DWORD *v60; // r11
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  int v64; // r11d
  _QWORD *v65; // rax
  _DWORD *v66; // [rsp+8h] [rbp-E8h]
  _QWORD *v67; // [rsp+8h] [rbp-E8h]
  _DWORD *v68; // [rsp+8h] [rbp-E8h]
  _QWORD *v69; // [rsp+10h] [rbp-E0h]
  _DWORD *v70; // [rsp+10h] [rbp-E0h]
  _DWORD *v71; // [rsp+10h] [rbp-E0h]
  _DWORD *v72; // [rsp+10h] [rbp-E0h]
  _DWORD *v73; // [rsp+18h] [rbp-D8h]
  __int64 v74; // [rsp+18h] [rbp-D8h]
  __int64 v75; // [rsp+18h] [rbp-D8h]
  __int64 v76; // [rsp+18h] [rbp-D8h]
  _DWORD *v77; // [rsp+18h] [rbp-D8h]
  _DWORD *v78; // [rsp+18h] [rbp-D8h]
  _DWORD *v79; // [rsp+20h] [rbp-D0h]
  __int64 v80; // [rsp+20h] [rbp-D0h]
  _QWORD *v81; // [rsp+20h] [rbp-D0h]
  __int64 v82; // [rsp+20h] [rbp-D0h]
  __int64 v83; // [rsp+20h] [rbp-D0h]
  __int64 v84; // [rsp+20h] [rbp-D0h]
  _DWORD *v85; // [rsp+20h] [rbp-D0h]
  _DWORD *v86; // [rsp+20h] [rbp-D0h]
  int v87; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v88; // [rsp+28h] [rbp-C8h]
  _DWORD *v89; // [rsp+28h] [rbp-C8h]
  __int32 v90; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v91; // [rsp+28h] [rbp-C8h]
  __int64 v96; // [rsp+58h] [rbp-98h]
  _QWORD *v97; // [rsp+58h] [rbp-98h]
  unsigned __int8 *v98; // [rsp+68h] [rbp-88h] BYREF
  unsigned __int8 *v99; // [rsp+70h] [rbp-80h] BYREF
  __int64 v100; // [rsp+78h] [rbp-78h]
  __int64 v101; // [rsp+80h] [rbp-70h]
  __m128i v102; // [rsp+90h] [rbp-60h] BYREF
  __int64 v103; // [rsp+A0h] [rbp-50h]
  __int64 v104; // [rsp+A8h] [rbp-48h]
  __int64 v105; // [rsp+B0h] [rbp-40h]

  v6 = a4 + 6;
  v8 = *(_QWORD **)a1;
  if ( *(_WORD *)(a2 + 68) == 3 )
  {
    v35 = v8[1] - 120LL;
    sub_2E32810((__int64 *)&v98, (__int64)a4, a4[7]);
    v99 = v98;
    if ( v98 )
    {
      sub_B976B0((__int64)&v98, v98, (__int64)&v99);
      v36 = (_QWORD *)a4[4];
      v98 = 0;
      v100 = 0;
      v101 = 0;
      v97 = v36;
      v102.m128i_i64[0] = (__int64)v99;
      if ( v99 )
        sub_B96E90((__int64)&v102, (__int64)v99, 1);
      v37 = sub_2E7B380(v97, v35, (unsigned __int8 **)&v102, 0);
    }
    else
    {
      v100 = 0;
      v101 = 0;
      v65 = (_QWORD *)a4[4];
      v102.m128i_i64[0] = 0;
      v97 = v65;
      v37 = sub_2E7B380(v65, v35, (unsigned __int8 **)&v102, 0);
    }
    v38 = (__int64)v37;
    if ( v102.m128i_i64[0] )
      sub_B91220((__int64)&v102, v102.m128i_i64[0]);
    sub_2E31040(a4 + 5, v38);
    v39 = a4[6];
    *(_QWORD *)(v38 + 8) = v6;
    *(_QWORD *)v38 = v39 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v38 & 7LL;
    *(_QWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v38;
    v40 = v100;
    a4[6] = v38 | a4[6] & 7LL;
    if ( v40 )
      sub_2E882B0(v38, (__int64)v97, v40);
    if ( v101 )
      sub_2E88680(v38, (__int64)v97, v101);
    v41 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 24LL);
    v102.m128i_i64[0] = 16;
    v103 = 0;
    LODWORD(v104) = v41;
    sub_2E8EAD0(v38, (__int64)v97, &v102);
    *(_DWORD *)(v38 + 44) = *(_DWORD *)(v38 + 44) & 0xC
                          | *(_DWORD *)(a2 + 44) & 0xFFFFF3
                          | *(_DWORD *)(v38 + 44) & 0xFF000000;
    if ( v99 )
      sub_B91220((__int64)&v99, (__int64)v99);
    if ( v98 )
      sub_B91220((__int64)&v98, (__int64)v98);
  }
  else
  {
    v9 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, _QWORD *, __int64))(*v8 + 248LL))(v8, a4, v6, a2);
    v12 = v9;
    if ( *(_BYTE *)(a1 + 56) )
    {
      v13 = *(_DWORD *)(v9 + 40) & 0xFFFFFF;
      if ( v13 )
      {
        v14 = 0;
        v96 = 40LL * v13;
        do
        {
          v21 = v14 + *(_QWORD *)(v12 + 32);
          if ( *(_BYTE *)v21 )
            goto LABEL_9;
          v22 = *(_DWORD *)(v21 + 8);
          if ( v22 >= 0 )
            goto LABEL_9;
          if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
          {
            v87 = sub_2EC06C0(
                    *(_QWORD *)(a1 + 24),
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL) + 16LL * (v22 & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL,
                    byte_3F871B3,
                    0,
                    v10,
                    v11);
            sub_2EAB0C0(v21, v87);
            v99 = (unsigned __int8 *)__PAIR64__(v87, v22);
            LODWORD(v100) = 0;
            sub_2FD83C0((__int64)&v102, a5, (int *)&v99, (__int64 *)((char *)&v99 + 4));
            if ( (unsigned __int8)sub_2FD5CE0(v22, a3, *(_QWORD *)(a1 + 24)) )
              goto LABEL_8;
            v15 = *(_QWORD *)(a6 + 8);
            v16 = *(_DWORD *)(a6 + 24);
            if ( v16 )
            {
              v17 = v16 - 1;
              v18 = 1;
              v19 = (v16 - 1) & (37 * v22);
              v20 = *(_DWORD *)(v15 + 4LL * v19);
              if ( v22 != v20 )
              {
                while ( v20 != -1 )
                {
                  v10 = (unsigned int)(v18 + 1);
                  v19 = v17 & (v18 + v19);
                  v20 = *(_DWORD *)(v15 + 4LL * v19);
                  if ( v22 == v20 )
                    goto LABEL_8;
                  ++v18;
                }
                goto LABEL_9;
              }
LABEL_8:
              sub_2FD7D90(a1, v22, v87, (__int64)a4);
            }
          }
          else
          {
            v23 = *(_QWORD *)(a5 + 8);
            v24 = *(unsigned int *)(a5 + 24);
            if ( (_DWORD)v24 )
            {
              v25 = (v24 - 1) & (37 * v22);
              v26 = (_DWORD *)(v23 + 12LL * v25);
              v27 = *v26;
              if ( v22 != *v26 )
              {
                v64 = 1;
                while ( v27 != -1 )
                {
                  v10 = (unsigned int)(v64 + 1);
                  v25 = (v24 - 1) & (v64 + v25);
                  v26 = (_DWORD *)(v23 + 12LL * v25);
                  v27 = *v26;
                  if ( v22 == *v26 )
                    goto LABEL_15;
                  v64 = v10;
                }
                goto LABEL_9;
              }
LABEL_15:
              if ( v26 != (_DWORD *)(v23 + 12 * v24) )
              {
                v28 = (unsigned int)v26[1];
                v29 = *(_QWORD *)(a1 + 24);
                v30 = *(_QWORD *)(v29 + 56);
                v31 = *(_QWORD *)(v30 + 16LL * (v22 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
                v32 = *(_QWORD *)(v30 + 16LL * (v26[1] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v26[2] )
                {
                  v79 = v26;
                  v88 = *(_QWORD *)(v30 + 16LL * (v22 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
                  v33 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, unsigned __int64))(**(_QWORD **)(a1 + 8)
                                                                                              + 256LL))(
                          *(_QWORD *)(a1 + 8),
                          v32,
                          v31);
                  v31 = v88;
                  v26 = v79;
                  if ( v33 )
                  {
                    sub_2EBE4E0(*(_QWORD *)(a1 + 24), v79[1], v33);
                    v26 = v79;
                    goto LABEL_37;
                  }
LABEL_42:
                  v73 = v26;
                  v43 = sub_2EC06C0(*(_QWORD *)(a1 + 24), v31, byte_3F871B3, 0, v31, v28);
                  v44 = *(unsigned __int8 **)(v12 + 56);
                  v45 = v73;
                  v90 = v43;
                  v46 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
                  v98 = v44;
                  v80 = v46 - 800;
                  if ( v44 )
                  {
                    sub_B96E90((__int64)&v98, (__int64)v44, 1);
                    v45 = v73;
                    v99 = v98;
                    if ( v98 )
                    {
                      sub_B976B0((__int64)&v98, v98, (__int64)&v99);
                      v45 = v73;
                      v98 = 0;
                    }
                  }
                  else
                  {
                    v99 = 0;
                  }
                  v100 = 0;
                  v101 = 0;
                  if ( (*(_BYTE *)(v12 + 44) & 4) != 0 )
                  {
                    v102.m128i_i64[0] = (__int64)v99;
                    v47 = (_QWORD *)a4[4];
                    if ( v99 )
                    {
                      v66 = v45;
                      v69 = (_QWORD *)a4[4];
                      sub_B96E90((__int64)&v102, (__int64)v99, 1);
                      v45 = v66;
                      v47 = v69;
                    }
                    v48 = v80;
                    v70 = v45;
                    v81 = v47;
                    v49 = sub_2E7B380(v47, v48, (unsigned __int8 **)&v102, 0);
                    v50 = (__int64)v81;
                    v51 = v70;
                    v52 = (__int64)v49;
                    if ( v102.m128i_i64[0] )
                    {
                      v67 = v49;
                      sub_B91220((__int64)&v102, v102.m128i_i64[0]);
                      v52 = (__int64)v67;
                      v51 = v70;
                      v50 = (__int64)v81;
                    }
                    v71 = v51;
                    v74 = v50;
                    v82 = v52;
                    sub_2E326B0((__int64)a4, (__int64 *)v12, v52);
                    v53 = v82;
                    v54 = v74;
                    v55 = v71;
                    if ( v100 )
                    {
                      sub_2E882B0(v82, v74, v100);
                      v55 = v71;
                      v54 = v74;
                      v53 = v82;
                    }
                    if ( v101 )
                    {
                      v72 = v55;
                      v75 = v54;
                      v83 = v53;
                      sub_2E88680(v53, v54, v101);
                      v55 = v72;
                      v54 = v75;
                      v53 = v83;
                    }
                    v68 = v55;
                    v102.m128i_i64[0] = 0x10000000;
                    v76 = v54;
                    v84 = v53;
                    v103 = 0;
                    v102.m128i_i32[2] = v90;
                    v104 = 0;
                    v105 = 0;
                    sub_2E8EAD0(v53, v54, &v102);
                    v56 = v84;
                    v57 = v76;
                    v58 = v68;
                  }
                  else
                  {
                    v78 = v45;
                    v62 = sub_2F26260((__int64)a4, (__int64 *)v12, (__int64 *)&v99, v80, v90);
                    v58 = v78;
                    v57 = (__int64)v62;
                    v56 = v63;
                  }
                  v77 = v58;
                  v59 = v58[2] & 0xFFF;
                  v102.m128i_i32[2] = v58[1];
                  v103 = 0;
                  v104 = 0;
                  v105 = 0;
                  v102.m128i_i64[0] = (unsigned int)(v59 << 8);
                  sub_2E8EAD0(v56, v57, &v102);
                  v60 = v77;
                  if ( v99 )
                  {
                    sub_B91220((__int64)&v99, (__int64)v99);
                    v60 = v77;
                  }
                  if ( v98 )
                  {
                    v85 = v60;
                    sub_B91220((__int64)&v98, (__int64)v98);
                    v60 = v85;
                  }
                  *v60 = -2;
                  LODWORD(v99) = v22;
                  --*(_DWORD *)(a5 + 16);
                  ++*(_DWORD *)(a5 + 20);
                  HIDWORD(v99) = v90;
                  LODWORD(v100) = 0;
                  sub_2FD83C0((__int64)&v102, a5, (int *)&v99, (__int64 *)((char *)&v99 + 4));
                  sub_2EAB0C0(v21, v90);
                }
                else
                {
                  if ( (unsigned __int16)(*(_WORD *)(v12 + 68) - 14) > 4u )
                  {
                    v86 = v26;
                    v91 = *(_QWORD *)(v30 + 16LL * (v22 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
                    v61 = sub_2EBE590(v29, v28, v31, 0);
                    v26 = v86;
                    v31 = v91;
                    v32 = v61;
                  }
                  if ( !v32 )
                    goto LABEL_42;
LABEL_37:
                  v89 = v26;
                  sub_2EAB0C0(v21, v26[1]);
                  v42 = (*(_DWORD *)v21 >> 8) & 0xFFF;
                  if ( v89[2] )
                  {
                    if ( v42 )
                      v42 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 296LL))(*(_QWORD *)(a1 + 8))
                          & 0xFFF;
                    else
                      v42 = v89[2] & 0xFFF;
                  }
                  *(_DWORD *)v21 = *(_DWORD *)v21 & 0xFFF000FF | (v42 << 8);
                }
                *(_BYTE *)(v21 + 3) &= ~0x40u;
              }
            }
          }
LABEL_9:
          v14 += 40;
        }
        while ( v14 != v96 );
      }
    }
  }
}
