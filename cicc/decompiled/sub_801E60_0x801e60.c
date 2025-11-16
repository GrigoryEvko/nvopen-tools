// Function: sub_801E60
// Address: 0x801e60
//
void __fastcall sub_801E60(__int64 a1, __int64 a2, __m128i *a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  char v5; // al
  int v6; // edx
  __int64 v7; // r13
  __m128i v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i *v12; // r14
  __m128i *v13; // r15
  __m128i *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __m128i v23; // xmm0
  __int64 v24; // r15
  __m128i *v25; // r14
  __int64 v26; // r13
  bool v27; // zf
  __int64 v28; // r9
  __m128i *v29; // r13
  __int64 v30; // r14
  __int64 v31; // rsi
  char v32; // al
  __int64 v33; // r8
  __int64 v34; // rdx
  bool v35; // cf
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int64 v40; // rdx
  const __m128i *i; // rdi
  __m128i *v42; // r14
  __m128i *v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rsi
  __int64 v46; // r12
  _QWORD *v47; // r15
  _BYTE *v48; // r13
  void *v49; // r13
  _QWORD *v50; // rax
  __int32 v51; // eax
  _BYTE *v52; // r15
  _QWORD *v53; // r15
  _QWORD *v54; // rax
  _QWORD *v55; // rbx
  __int64 v56; // r15
  _QWORD *v57; // rax
  __int64 v58; // r13
  _QWORD *v59; // rax
  __int64 v60; // rax
  __int64 v61; // r9
  __m128i *v62; // r8
  __m128i *v63; // rax
  __int64 v64; // [rsp+0h] [rbp-160h]
  __int64 v65; // [rsp+8h] [rbp-158h]
  __m128i *v66; // [rsp+10h] [rbp-150h]
  __int64 v67; // [rsp+18h] [rbp-148h]
  __int64 v69; // [rsp+28h] [rbp-138h]
  __m128i *v70; // [rsp+28h] [rbp-138h]
  __m128i *v72; // [rsp+38h] [rbp-128h]
  char v73; // [rsp+57h] [rbp-109h]
  __m128i *v74; // [rsp+58h] [rbp-108h]
  __m128i *v75; // [rsp+58h] [rbp-108h]
  __int64 v76; // [rsp+58h] [rbp-108h]
  __int64 v77; // [rsp+58h] [rbp-108h]
  __int64 v78; // [rsp+58h] [rbp-108h]
  __int64 v79; // [rsp+68h] [rbp-F8h] BYREF
  __m128i *v80; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v81; // [rsp+78h] [rbp-E8h]
  int v82; // [rsp+80h] [rbp-E0h]
  int v83; // [rsp+84h] [rbp-DCh]
  __m128i v84; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int64 v85; // [rsp+A0h] [rbp-C0h]
  __m128i v86; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned __int64 v87; // [rsp+C0h] [rbp-A0h]
  __int64 v88; // [rsp+C8h] [rbp-98h]
  __int64 v89; // [rsp+D0h] [rbp-90h]
  unsigned int v90; // [rsp+D8h] [rbp-88h]
  __m128i v91; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v92; // [rsp+F0h] [rbp-70h]
  const __m128i *v93; // [rsp+F8h] [rbp-68h]
  int v94; // [rsp+108h] [rbp-58h]

  v3 = a1;
  v4 = *(_QWORD *)(a1 + 128);
  v5 = *(_BYTE *)(v4 + 140);
  v79 = 0;
  v73 = v5;
  if ( v5 == 12 )
  {
    do
      v4 = *(_QWORD *)(v4 + 160);
    while ( *(_BYTE *)(v4 + 140) == 12 );
    v73 = *(_BYTE *)(v4 + 140);
  }
  if ( dword_4F077C4 == 2 && (unsigned __int8)(v73 - 9) <= 2u )
    sub_7E3EE0(v4);
  sub_7F51D0(*(_QWORD *)(a1 + 176), 1, 0, (__int64)&v80);
  if ( a2 )
  {
    v7 = *(_QWORD *)(a2 + 176);
    sub_7F51D0(v7, 1, v6, (__int64)&v84);
    v72 = 0;
    if ( v73 == 11 )
    {
      v60 = sub_72FD90(*(_QWORD *)(v4 + 160), 11);
      v61 = 0;
      if ( v7 && *(_BYTE *)(v7 + 173) == 13 )
      {
        if ( *(_QWORD *)(v7 + 184) != v60 )
          v61 = v7;
        if ( v84.m128i_i64[1] )
        {
          --v84.m128i_i64[1];
        }
        else if ( v84.m128i_i64[0] )
        {
          v78 = v60;
          sub_7F51D0(*(_QWORD *)(v84.m128i_i64[0] + 120), v85, SHIDWORD(v85), (__int64)&v84);
          v60 = v78;
        }
        v7 = *(_QWORD *)(v7 + 120);
      }
      v72 = v80;
      if ( v80 )
      {
        if ( v80[10].m128i_i8[13] == 13 )
        {
          v62 = 0;
          if ( v80[11].m128i_i64[1] != v60 )
            v62 = v80;
          if ( v81 )
          {
            --v81;
          }
          else
          {
            sub_7F51D0(v80[7].m128i_i64[1], v82, v83, (__int64)&v80);
            v72 = v80;
          }
          v63 = v72;
          v72 = v62;
          *(_QWORD *)(a1 + 176) = v63;
          sub_7F6F90(v7, v61, (__int64)v62, (__int64)&v84, &v79, v61);
          goto LABEL_7;
        }
        v72 = 0;
      }
      sub_7F6F90(v7, v61, (__int64)v72, (__int64)&v84, &v79, v61);
    }
  }
  else
  {
    sub_7F51D0(0, 1, 0, (__int64)&v84);
    v72 = 0;
  }
LABEL_7:
  v12 = v80;
  v13 = 0;
  v14 = 0;
  v74 = 0;
  while ( 1 )
  {
LABEL_8:
    if ( v12 )
    {
      while ( 1 )
      {
        v8.m128i_i8[0] = v12[10].m128i_i8[13];
        if ( v8.m128i_i8[0] == 11 )
        {
          v30 = v12[11].m128i_i64[0];
          v31 = sub_8D4050(v4);
          if ( !(unsigned int)sub_8D44E0(v30, v31) )
          {
            sub_7F7440((__int64)v80, v31);
            v30 = v80[11].m128i_i64[0];
            v81 = v80[11].m128i_i64[1];
          }
          v32 = *(_BYTE *)(v30 + 173);
          if ( v32 == 13 || v32 == 10 )
            sub_7F66A0(&v80);
          v12 = v80;
          v8.m128i_i8[0] = v80[10].m128i_i8[13];
        }
        v15 = v84.m128i_i64[0];
        if ( v8.m128i_i8[0] == 13 )
          break;
        if ( v14 != v12 )
        {
          v74 = 0;
          v14 = 0;
        }
        if ( v84.m128i_i64[0] && v8.m128i_i8[0] != 2 )
        {
          sub_740640(v84.m128i_i64[0]);
          v12 = v80;
          v8.m128i_i8[0] = v80[10].m128i_i8[13];
        }
        if ( v8.m128i_i8[0] == 10 )
        {
          v24 = v84.m128i_i64[0];
          if ( v84.m128i_i64[0] )
          {
            sub_7F66A0((__m128i **)&v84);
            v24 = v84.m128i_i64[0];
            v12 = v80;
            if ( *(_BYTE *)(v84.m128i_i64[0] + 173) == 9 )
            {
              v24 = 0;
            }
            else if ( (v80[10].m128i_i8[9] & 0x20) != 0 )
            {
              v24 = 0;
            }
          }
          sub_801E60(v12, v24, a3);
          if ( v84.m128i_i64[0] == v24 )
            goto LABEL_21;
          v25 = v80;
          if ( !HIDWORD(qword_4F077B4) )
          {
            if ( (v80[10].m128i_i16[5] & 0x4040) != 0 )
              goto LABEL_80;
            goto LABEL_81;
          }
          v11 = (unsigned int)qword_4F077B4;
          if ( !(_DWORD)qword_4F077B4 || (v80[10].m128i_i16[5] & 0x4040) == 0 )
            goto LABEL_81;
LABEL_80:
          if ( *(_BYTE *)(v84.m128i_i64[0] + 173) != 9 )
          {
LABEL_81:
            sub_7408F0(v84.m128i_i64[0], (__int64)v80, v8.m128i_i64[1], v9, v10, v11);
            goto LABEL_21;
          }
          v67 = *(_QWORD *)(v84.m128i_i64[0] + 176);
          v70 = sub_7E7CA0(*(_QWORD *)(v84.m128i_i64[0] + 128));
          sub_7F9080((__int64)v70, (__int64)&v91);
          sub_7FEC50(v67, &v91, 0, (__int64)v70, 0, 0, a3, 0, 0);
          v45 = 11;
          sub_7F5D80(v25[8].m128i_i64[0], 11, (__int64)&v86);
          if ( !v25[11].m128i_i64[0] )
            goto LABEL_157;
          v66 = v14;
          v65 = v4;
          v64 = v3;
          v46 = v25[11].m128i_i64[0];
          v47 = (_QWORD *)v67;
          while ( 2 )
          {
            if ( (*(_BYTE *)(v46 + 171) & 0x40) != 0 )
            {
              if ( v86.m128i_i32[0] )
              {
                v48 = sub_7E2510((__int64)v70, v45);
                *((_QWORD *)v48 + 2) = sub_73A830(v87, byte_4F06A51[0]);
                v49 = sub_73DBF0(0x5Cu, v88, (__int64)v48);
              }
              else
              {
                v58 = v86.m128i_i64[1];
                v59 = sub_73E830((__int64)v70);
                v49 = sub_73E8A0((__int64)v59, v58);
              }
              goto LABEL_137;
            }
            if ( *(_BYTE *)(v46 + 173) == 10 )
            {
              v47 = sub_725A70(2u);
              v47[7] = sub_7401F0(v46);
            }
            else
            {
              v49 = (void *)*sub_740760(v46, v45);
LABEL_137:
              if ( v49 )
              {
                v50 = sub_725A70(3u);
                v50[7] = v49;
                v47 = v50;
              }
            }
            v45 = 9;
            sub_724A80(v46, 9);
            *(_QWORD *)(v46 + 176) = v47;
            if ( v86.m128i_i32[0] )
            {
              v8.m128i_i64[1] = v87;
              if ( v87 >= v89 - 1 )
                goto LABEL_141;
LABEL_154:
              ++v8.m128i_i64[1];
              v8.m128i_i32[0] = 1;
              v87 = v8.m128i_u64[1];
            }
            else if ( v86.m128i_i64[1] && (v45 = 11, sub_72FD90(*(_QWORD *)(v86.m128i_i64[1] + 112), 11)) )
            {
              v45 = v86.m128i_u32[0];
              if ( v86.m128i_i32[0] )
              {
                v8.m128i_i64[1] = v87;
                goto LABEL_154;
              }
              v45 = v90;
              v86.m128i_i64[1] = sub_72FD90(*(_QWORD *)(v86.m128i_i64[1] + 112), v90);
              v88 = *(_QWORD *)(v86.m128i_i64[1] + 120);
              v8.m128i_i32[0] = 1;
            }
            else
            {
LABEL_141:
              v8.m128i_i32[0] = 0;
            }
            v46 = *(_QWORD *)(v46 + 120);
            if ( !v46 )
            {
              v14 = v66;
              v3 = v64;
              if ( !v8.m128i_i32[0] )
                goto LABEL_157;
              v51 = v86.m128i_i32[0];
LABEL_145:
              if ( v51 )
              {
LABEL_146:
                v52 = sub_7E2510((__int64)v70, v45);
                *((_QWORD *)v52 + 2) = sub_73A830(v87, byte_4F06A51[0]);
                v45 = v88;
                v53 = sub_73DBF0(0x5Cu, v88, (__int64)v52);
                goto LABEL_147;
              }
              while ( 1 )
              {
                v56 = v86.m128i_i64[1];
                v57 = sub_73E830((__int64)v70);
                v45 = v56;
                v53 = sub_73E8A0((__int64)v57, v56);
LABEL_147:
                v54 = sub_725A70(3u);
                v54[7] = v53;
                v55 = v54;
                v8.m128i_i64[0] = (__int64)sub_724D50(9);
                *(_QWORD *)(v8.m128i_i64[0] + 176) = v55;
                *(_QWORD *)(v8.m128i_i64[0] + 128) = *v53;
                v8.m128i_i64[1] = v25[11].m128i_i64[1];
                if ( v8.m128i_i64[1] )
                  *(_QWORD *)(v8.m128i_i64[1] + 120) = v8.m128i_i64[0];
                v25[11].m128i_i64[1] = v8.m128i_i64[0];
                v8.m128i_i32[0] = v86.m128i_i32[0];
                if ( v86.m128i_i32[0] )
                {
                  v8.m128i_i64[1] = v87;
                  v9 = v89 - 1;
                  if ( v87 >= v89 - 1 )
                    goto LABEL_156;
                }
                else
                {
                  if ( !v86.m128i_i64[1] || (v45 = 11, !sub_72FD90(*(_QWORD *)(v86.m128i_i64[1] + 112), 11)) )
                  {
LABEL_156:
                    v4 = v65;
                    v25[10].m128i_i8[10] &= ~0x40u;
LABEL_157:
                    v25[10].m128i_i8[11] &= ~0x40u;
LABEL_21:
                    v20 = v84.m128i_i64[1];
                    if ( !v84.m128i_i64[1] )
                    {
                      if ( v84.m128i_i64[0] )
                        sub_7F51D0(*(_QWORD *)(v84.m128i_i64[0] + 120), v85, SHIDWORD(v85), (__int64)&v84);
                      goto LABEL_24;
                    }
LABEL_64:
                    v84.m128i_i64[1] = v20 - 1;
                    goto LABEL_24;
                  }
                  v8.m128i_i32[0] = v86.m128i_i32[0];
                  if ( !v86.m128i_i32[0] )
                  {
                    v45 = v90;
                    v86.m128i_i64[1] = sub_72FD90(*(_QWORD *)(v86.m128i_i64[1] + 112), v90);
                    v88 = *(_QWORD *)(v86.m128i_i64[1] + 120);
                    v51 = v86.m128i_i32[0];
                    goto LABEL_145;
                  }
                  v8.m128i_i64[1] = v87;
                }
                v87 = v8.m128i_i64[1] + 1;
                if ( v8.m128i_i32[0] )
                  goto LABEL_146;
              }
            }
            continue;
          }
        }
        if ( v8.m128i_i8[0] == 9 )
          sub_802F60(v12[11].m128i_i64[0], v12[8].m128i_i64[0]);
        if ( v84.m128i_i64[0] )
        {
          sub_7F66A0((__m128i **)&v84);
          sub_7F66A0(&v80);
          sub_7408F0(v84.m128i_i64[0], (__int64)v80, v16, v17, v18, v19);
          goto LABEL_21;
        }
        v20 = v84.m128i_i64[1];
        if ( v84.m128i_i64[1] )
          goto LABEL_64;
LABEL_24:
        v13 = v80;
        if ( !v81 )
        {
          v12 = 0;
          if ( !v80 )
            goto LABEL_26;
          sub_7F51D0(v80[7].m128i_i64[1], v82, v83, (__int64)&v80);
          v12 = v80;
          goto LABEL_8;
        }
        v12 = v80;
        --v81;
        if ( !v80 )
          goto LABEL_26;
      }
      if ( v13 )
      {
        v13[7].m128i_i64[1] = 0;
        *(_QWORD *)(v3 + 184) = v13;
        if ( !v15 )
          goto LABEL_39;
        goto LABEL_34;
      }
      *(_QWORD *)(v3 + 176) = 0;
      *(_QWORD *)(v3 + 184) = 0;
      if ( !v15 )
        goto LABEL_39;
      v13 = v12;
    }
    else
    {
LABEL_26:
      v15 = v84.m128i_i64[0];
      *(_QWORD *)(v3 + 184) = v13;
      if ( !v15 )
        break;
      if ( v13 )
      {
LABEL_34:
        v13[7].m128i_i64[1] = v15;
        goto LABEL_35;
      }
    }
    *(_QWORD *)(v3 + 176) = v15;
    v12 = v13;
LABEL_35:
    v21 = *(_QWORD *)(v15 + 120);
    if ( v21 )
    {
      do
      {
        v15 = v21;
        v21 = *(_QWORD *)(v21 + 120);
      }
      while ( v21 );
      v84.m128i_i64[0] = v15;
    }
    *(_QWORD *)(v3 + 184) = v15;
    if ( !v12 )
      break;
LABEL_39:
    if ( v73 != 11 )
    {
      if ( *(_BYTE *)(v4 + 140) == 8 )
      {
        if ( v74 )
        {
          if ( v14 )
          {
            if ( v14[10].m128i_i8[13] != 11 )
            {
              v37 = v12[11].m128i_u64[1];
              v38 = v74[11].m128i_u64[1];
              if ( v37 > v38 )
              {
                v39 = v14[7].m128i_i64[1];
                v40 = ~v38 + v37;
                if ( !v40 )
                {
                  v13 = v14;
                  sub_7F51D0(v39, 1, 0, (__int64)&v84);
                  goto LABEL_57;
                }
                if ( !(a2 | v39) )
                {
                  v77 = v40;
                  for ( i = (const __m128i *)sub_8D4050(v4);
                        i[8].m128i_i8[12] == 12;
                        i = (const __m128i *)i[10].m128i_i64[0] )
                  {
                    ;
                  }
                  v13 = (__m128i *)sub_7F6BA0(i);
                  v42 = v13;
                  if ( v77 != 1 )
                  {
                    v43 = (__m128i *)sub_724D50(11);
                    v43[11].m128i_i64[0] = (__int64)v13;
                    v13 = v43;
                    v43[11].m128i_i64[1] = v77;
                    if ( (v42[-1].m128i_i8[8] & 8) == 0 )
                      v43[-1].m128i_i8[8] &= ~8u;
                  }
                  v14[7].m128i_i64[1] = (__int64)v13;
                  sub_7F51D0(0, 1, 0, (__int64)&v84);
                  goto LABEL_57;
                }
              }
            }
          }
        }
      }
      v13 = 0;
      sub_7F5D80(*(_QWORD *)(v3 + 128), 3, (__int64)&v91);
      sub_7F51D0(*(_QWORD *)(v3 + 176), 1, 0, (__int64)&v86);
      v10 = v86.m128i_i64[0];
LABEL_42:
      v9 = v91.m128i_u32[0];
      v8.m128i_i64[0] = v12[11].m128i_i64[1];
      if ( v91.m128i_i32[0] )
      {
LABEL_43:
        if ( v8.m128i_i64[0] == v92 )
          goto LABEL_53;
        goto LABEL_44;
      }
      while ( 1 )
      {
        if ( v8.m128i_i64[0] == v91.m128i_i64[1] )
        {
LABEL_53:
          v75 = (__m128i *)v10;
          if ( v10 )
          {
            sub_7F66A0((__m128i **)&v86);
            v10 = (__int64)v75;
            if ( (__m128i *)v86.m128i_i64[0] != v75 )
              v13 = v75;
          }
          v23 = _mm_loadu_si128(&v86);
          v85 = v87;
          v84 = v23;
          goto LABEL_57;
        }
LABEL_44:
        if ( v10 )
          goto LABEL_45;
        v33 = sub_7F6BA0(v93);
        if ( v91.m128i_i64[1] && (*(_BYTE *)(v91.m128i_i64[1] + 146) & 8) != 0 )
        {
          *(_BYTE *)(v33 + 171) |= 0x20u;
          *(_BYTE *)(v3 + 171) |= 0x40u;
          if ( v13 )
          {
LABEL_93:
            v13[7].m128i_i64[1] = v33;
            goto LABEL_94;
          }
        }
        else
        {
          *(_BYTE *)(v3 + 171) |= 0x40u;
          if ( v13 )
            goto LABEL_93;
        }
        *(_QWORD *)(v3 + 176) = v33;
LABEL_94:
        v76 = v33;
        sub_7F51D0(v33, 1, 0, (__int64)&v86);
        if ( !v91.m128i_i32[0] )
        {
          v8.m128i_i64[0] = v86.m128i_i64[1];
          if ( !v86.m128i_i64[1] )
            goto LABEL_99;
          goto LABEL_46;
        }
        v22 = v92;
        v34 = v12[11].m128i_i64[1];
        v35 = v34 == v92;
        v8.m128i_i64[1] = v34 - v92;
        if ( !v35 && v8.m128i_i64[1] != 1 )
        {
          v69 = v8.m128i_i64[1];
          v44 = sub_724D50(11);
          *((_QWORD *)v44 + 23) = v69;
          *((_QWORD *)v44 + 22) = v76;
          if ( (*(_BYTE *)(v76 - 8) & 8) == 0 )
            *(v44 - 8) &= ~8u;
          sub_7F51D0((__int64)v44, 1, 0, (__int64)&v86);
          if ( v13 )
            v13[7].m128i_i64[1] = v11;
          else
            *(_QWORD *)(v3 + 176) = v11;
LABEL_45:
          v8.m128i_i64[0] = v86.m128i_i64[1];
          if ( !v86.m128i_i64[1] )
          {
            if ( v91.m128i_i32[0] )
            {
              v22 = v92;
LABEL_67:
              v8.m128i_i64[0] = v86.m128i_i64[0];
              v92 = v22 + 1;
              v13 = (__m128i *)v86.m128i_i64[0];
            }
            else
            {
LABEL_99:
              v36 = sub_72FD90(*(_QWORD *)(v91.m128i_i64[1] + 112), v94);
              v13 = (__m128i *)v86.m128i_i64[0];
              v91.m128i_i64[1] = v36;
              v93 = *(const __m128i **)(v36 + 120);
              v8 = v86;
              if ( v86.m128i_i64[1] )
              {
                v8.m128i_i64[1] = v86.m128i_i64[1] - 1;
                v10 = v86.m128i_i64[0];
                --v86.m128i_i64[1];
                goto LABEL_42;
              }
            }
            v10 = 0;
            if ( v8.m128i_i64[0] )
            {
              sub_7F51D0(*(_QWORD *)(v8.m128i_i64[0] + 120), v87, SHIDWORD(v87), (__int64)&v86);
              v10 = v86.m128i_i64[0];
            }
            goto LABEL_42;
          }
LABEL_46:
          v22 = v92;
          v8.m128i_i64[1] = v12[11].m128i_i64[1] - v92;
          goto LABEL_47;
        }
        v8.m128i_i64[0] = v86.m128i_i64[1];
        if ( !v86.m128i_i64[1] )
          goto LABEL_67;
LABEL_47:
        v10 = v86.m128i_i64[0];
        if ( v8.m128i_i64[1] > (unsigned __int64)v8.m128i_i64[0] )
          v8.m128i_i64[1] = v8.m128i_i64[0];
        v92 = v8.m128i_i64[1] + v22;
        v86.m128i_i64[1] = v8.m128i_i64[0] - v8.m128i_i64[1];
        if ( v8.m128i_i64[0] != v8.m128i_i64[1] )
          goto LABEL_42;
        v13 = 0;
        if ( !v86.m128i_i64[0] )
          goto LABEL_42;
        v13 = (__m128i *)v86.m128i_i64[0];
        sub_7F51D0(*(_QWORD *)(v86.m128i_i64[0] + 120), v87, SHIDWORD(v87), (__int64)&v86);
        v9 = v91.m128i_u32[0];
        v10 = v86.m128i_i64[0];
        v8.m128i_i64[0] = v12[11].m128i_i64[1];
        if ( v91.m128i_i32[0] )
          goto LABEL_43;
      }
    }
    v26 = v12[11].m128i_i64[1];
    v27 = v26 == sub_72FD90(*(_QWORD *)(v4 + 160), 11);
    v29 = 0;
    if ( !v27 )
      v29 = v80;
    v13 = 0;
    sub_7F6F90(*(_QWORD *)(v3 + 176), (__int64)v72, (__int64)v29, (__int64)&v84, &v79, v28);
    v72 = v29;
    *(_QWORD *)(v3 + 176) = 0;
LABEL_57:
    v14 = v80;
    v74 = v80;
    if ( v81 )
    {
      --v81;
      goto LABEL_59;
    }
    v14 = 0;
    if ( v80 )
    {
      sub_7F51D0(v80[7].m128i_i64[1], v82, v83, (__int64)&v80);
      v14 = v80;
      if ( !v13 )
        goto LABEL_72;
LABEL_60:
      v13[7].m128i_i64[1] = (__int64)v14;
      v12 = v14;
    }
    else
    {
LABEL_59:
      if ( v13 )
        goto LABEL_60;
LABEL_72:
      *(_QWORD *)(v3 + 176) = v14;
      v12 = v14;
    }
  }
  if ( v79 )
    sub_7408F0(v79, *(_QWORD *)(v3 + 176), v8.m128i_i64[1], v9, v10, v11);
  if ( v72 )
  {
    v72[7].m128i_i64[1] = *(_QWORD *)(v3 + 176);
    *(_QWORD *)(v3 + 176) = v72;
  }
}
