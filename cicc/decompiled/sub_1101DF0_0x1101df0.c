// Function: sub_1101DF0
// Address: 0x1101df0
//
unsigned __int8 *__fastcall sub_1101DF0(const __m128i *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v4; // r8
  __int64 v5; // r14
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // r13
  __int64 v9; // rdx
  __int16 v10; // r15
  __int64 v11; // rax
  unsigned __int8 *v12; // r14
  unsigned int v14; // ebx
  int v15; // eax
  int v16; // eax
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  unsigned __int64 v19; // xmm2_8
  __m128i v20; // xmm3
  unsigned __int32 v21; // ebx
  unsigned __int8 *v22; // r8
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned __int64 v25; // rax
  const void **v26; // rsi
  __int64 v27; // r13
  unsigned int v28; // edx
  int v29; // eax
  bool v30; // al
  __int64 v31; // rdi
  __int64 v32; // rdx
  unsigned int v33; // r14d
  int v34; // eax
  bool v35; // al
  __int64 v36; // r14
  int v37; // eax
  __int64 v38; // rax
  unsigned int **v39; // r15
  _BYTE *v40; // r14
  __m128i v41; // rax
  __int64 v42; // r14
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // r13
  unsigned int v46; // eax
  unsigned int v47; // r10d
  __int64 v48; // r13
  _BYTE *v49; // rax
  unsigned int v50; // r14d
  int v51; // eax
  int v52; // eax
  unsigned int v53; // r14d
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  __int64 v57; // rax
  __int64 v58; // rdx
  int v59; // r14d
  __int64 v60; // rbx
  __int64 v61; // r14
  __int64 v62; // rdx
  unsigned int v63; // esi
  unsigned __int8 *v64; // [rsp+8h] [rbp-E8h]
  unsigned int v65; // [rsp+10h] [rbp-E0h]
  char v66; // [rsp+10h] [rbp-E0h]
  int v67; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v68; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v69; // [rsp+18h] [rbp-D8h]
  unsigned int v70; // [rsp+18h] [rbp-D8h]
  int v71; // [rsp+18h] [rbp-D8h]
  int v72; // [rsp+18h] [rbp-D8h]
  __int64 v73; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v74; // [rsp+20h] [rbp-D0h]
  __int64 v75; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v76; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v77; // [rsp+20h] [rbp-D0h]
  __int64 v79; // [rsp+30h] [rbp-C0h] BYREF
  __int32 v80; // [rsp+38h] [rbp-B8h]
  __int64 v81; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int32 v82; // [rsp+48h] [rbp-A8h]
  __int64 v83; // [rsp+50h] [rbp-A0h]
  unsigned int v84; // [rsp+58h] [rbp-98h]
  __int16 v85; // [rsp+60h] [rbp-90h]
  __m128i v86; // [rsp+70h] [rbp-80h] BYREF
  __m128i v87; // [rsp+80h] [rbp-70h]
  unsigned __int64 v88; // [rsp+90h] [rbp-60h]
  __int64 v89; // [rsp+98h] [rbp-58h]
  __m128i v90; // [rsp+A0h] [rbp-50h]
  __int64 v91; // [rsp+B0h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 32);
  v5 = *((_QWORD *)v4 + 1);
  v6 = *(unsigned __int8 *)(v5 + 8);
  v7 = v6 - 17;
  if ( (unsigned int)(v6 - 17) <= 1 )
    LOBYTE(v6) = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
  if ( (_BYTE)v6 != 12 )
    return 0;
  v8 = *(_QWORD *)(a2 - 64);
  v9 = *v4;
  v10 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v10 != 40 )
    goto LABEL_5;
  if ( (_BYTE)v9 == 17 )
  {
    v33 = *((_DWORD *)v4 + 8);
    if ( v33 <= 0x40 )
    {
      v35 = *((_QWORD *)v4 + 3) == 0;
    }
    else
    {
      v74 = *(unsigned __int8 **)(a2 - 32);
      v34 = sub_C444A0((__int64)(v4 + 24));
      v4 = v74;
      v35 = v33 == v34;
    }
  }
  else
  {
    if ( v7 > 1 || (unsigned __int8)v9 > 0x15u )
      return 0;
    v76 = *(unsigned __int8 **)(a2 - 32);
    v49 = sub_AD7630((__int64)v4, 0, v9);
    v4 = v76;
    if ( !v49 || *v49 != 17 )
    {
      if ( *(_BYTE *)(v5 + 8) == 17 )
      {
        v52 = *(_DWORD *)(v5 + 32);
        v66 = 0;
        v53 = 0;
        v72 = v52;
        if ( v52 )
        {
          while ( 1 )
          {
            v77 = v4;
            v54 = sub_AD69F0(v4, v53);
            v4 = v77;
            if ( !v54 )
              break;
            if ( *(_BYTE *)v54 != 13 )
            {
              if ( *(_BYTE *)v54 != 17 )
                goto LABEL_47;
              if ( *(_DWORD *)(v54 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v54 + 24) )
                  goto LABEL_47;
              }
              else
              {
                v67 = *(_DWORD *)(v54 + 32);
                v55 = sub_C444A0(v54 + 24);
                v4 = v77;
                if ( v67 != v55 )
                  goto LABEL_47;
              }
              v66 = 1;
            }
            if ( v72 == ++v53 )
            {
              if ( v66 )
                goto LABEL_41;
              goto LABEL_47;
            }
          }
        }
      }
      goto LABEL_47;
    }
    v50 = *((_DWORD *)v49 + 8);
    if ( v50 <= 0x40 )
    {
      v35 = *((_QWORD *)v49 + 3) == 0;
    }
    else
    {
      v51 = sub_C444A0((__int64)(v49 + 24));
      v4 = v76;
      v35 = v50 == v51;
    }
  }
  if ( !v35 )
  {
LABEL_47:
    LOBYTE(v9) = *v4;
LABEL_5:
    if ( (_BYTE)v9 == 17 )
    {
      v11 = *(_QWORD *)(a2 + 16);
      if ( v11 )
      {
        v12 = *(unsigned __int8 **)(v11 + 8);
        if ( !v12 && (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
        {
          v14 = *((_DWORD *)v4 + 8);
          v73 = (__int64)(v4 + 24);
          if ( v14 <= 0x40 )
          {
            v57 = *((_QWORD *)v4 + 3);
            if ( !v57 || (v57 & (v57 - 1)) == 0 )
            {
LABEL_14:
              v17 = _mm_loadu_si128(a1 + 6);
              v18 = _mm_loadu_si128(a1 + 7);
              v64 = v4;
              v19 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
              v91 = a1[10].m128i_i64[0];
              v20 = _mm_loadu_si128(a1 + 9);
              v86 = v17;
              v88 = v19;
              v87 = v18;
              v89 = a3;
              v90 = v20;
              sub_9AC330((__int64)&v81, v8, 0, &v86);
              v21 = v82;
              v22 = v64;
              v86.m128i_i32[2] = v82;
              if ( v82 > 0x40 )
              {
                v26 = (const void **)&v81;
                sub_C43780((__int64)&v86, (const void **)&v81);
                v21 = v86.m128i_u32[2];
                v22 = v64;
                if ( v86.m128i_i32[2] > 0x40u )
                {
                  sub_C43D10((__int64)&v86);
                  v21 = v86.m128i_u32[2];
                  v27 = v86.m128i_i64[0];
                  v22 = v64;
                  v80 = v86.m128i_i32[2];
                  v79 = v86.m128i_i64[0];
                  if ( v86.m128i_i32[2] > 0x40u )
                  {
                    v56 = sub_C44630((__int64)&v79);
                    v22 = v64;
                    if ( v56 != 1 )
                      goto LABEL_29;
                    goto LABEL_21;
                  }
LABEL_19:
                  if ( !v27 || (v27 & (v27 - 1)) != 0 )
                    goto LABEL_31;
LABEL_21:
                  v28 = *((_DWORD *)v22 + 8);
                  if ( v28 <= 0x40 )
                  {
                    v30 = *((_QWORD *)v22 + 3) == 0;
                  }
                  else
                  {
                    v65 = *((_DWORD *)v22 + 8);
                    v69 = v22;
                    v29 = sub_C444A0(v73);
                    v28 = v65;
                    v22 = v69;
                    v30 = v65 == v29;
                  }
                  if ( !v30 )
                  {
                    if ( v28 <= 0x40 )
                    {
                      if ( v27 != *((_QWORD *)v22 + 3) )
                      {
LABEL_26:
                        v31 = *(_QWORD *)(a3 + 8);
                        if ( v10 == 33 )
                          v32 = sub_AD62B0(v31);
                        else
                          v32 = sub_AD6530(v31, (__int64)v26);
                        v12 = sub_F162A0((__int64)a1, a3, v32);
                        if ( v21 > 0x40 )
                        {
LABEL_29:
                          if ( v27 )
                            j_j___libc_free_0_0(v27);
                        }
LABEL_31:
                        if ( v84 > 0x40 && v83 )
                          j_j___libc_free_0_0(v83);
                        if ( v82 > 0x40 )
                        {
                          if ( v81 )
                            j_j___libc_free_0_0(v81);
                        }
                        return v12;
                      }
                    }
                    else
                    {
                      v26 = (const void **)&v79;
                      if ( !sub_C43C50(v73, (const void **)&v79) )
                        goto LABEL_26;
                    }
                  }
                  if ( v21 > 0x40 )
                    goto LABEL_29;
                  goto LABEL_31;
                }
                v23 = v86.m128i_i64[0];
              }
              else
              {
                v23 = v81;
              }
              v80 = v21;
              v24 = ~v23;
              v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
              if ( !v21 )
                v25 = 0;
              v26 = (const void **)(v25 & v24);
              v79 = (__int64)v26;
              v27 = (__int64)v26;
              goto LABEL_19;
            }
          }
          else
          {
            v68 = v4;
            v15 = sub_C444A0(v73);
            v4 = v68;
            if ( v14 == v15 )
              goto LABEL_14;
            v16 = sub_C44630(v73);
            v4 = v68;
            if ( v16 == 1 )
              goto LABEL_14;
          }
        }
      }
    }
    return 0;
  }
LABEL_41:
  v36 = *(_QWORD *)(v8 + 8);
  v37 = sub_BCB060(v36);
  v38 = sub_AD64C0(v36, (unsigned int)(v37 - 1), 0);
  v39 = (unsigned int **)a1[2].m128i_i64[0];
  v40 = (_BYTE *)v38;
  v41.m128i_i64[0] = (__int64)sub_BD5D20(v8);
  LOWORD(v88) = 773;
  v86 = v41;
  v87.m128i_i64[0] = (__int64)".lobit";
  v42 = sub_920F70(v39, (_BYTE *)v8, v40, (__int64)&v86, 0);
  v43 = *(_QWORD *)(a3 + 8);
  if ( v43 != *(_QWORD *)(v42 + 8) )
  {
    v44 = a1[2].m128i_i64[0];
    v85 = 257;
    v45 = *(_QWORD *)(v42 + 8);
    v75 = v44;
    v70 = sub_BCB060(v45);
    v46 = sub_BCB060(v43);
    v47 = v46 < v70 ? 38 : 40;
    if ( v43 == v45 )
    {
      v48 = v42;
    }
    else
    {
      v71 = v46 < v70 ? 38 : 40;
      v48 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v75 + 80) + 120LL))(
              *(_QWORD *)(v75 + 80),
              v47,
              v42,
              v43);
      if ( !v48 )
      {
        LOWORD(v88) = 257;
        v48 = sub_B51D30(v71, v42, v43, (__int64)&v86, 0, 0);
        if ( (unsigned __int8)sub_920620(v48) )
        {
          v58 = *(_QWORD *)(v75 + 96);
          v59 = *(_DWORD *)(v75 + 104);
          if ( v58 )
            sub_B99FD0(v48, 3u, v58);
          sub_B45150(v48, v59);
        }
        (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v75 + 88) + 16LL))(
          *(_QWORD *)(v75 + 88),
          v48,
          &v81,
          *(_QWORD *)(v75 + 56),
          *(_QWORD *)(v75 + 64));
        v60 = *(_QWORD *)v75;
        v61 = *(_QWORD *)v75 + 16LL * *(unsigned int *)(v75 + 8);
        if ( *(_QWORD *)v75 != v61 )
        {
          do
          {
            v62 = *(_QWORD *)(v60 + 8);
            v63 = *(_DWORD *)v60;
            v60 += 16;
            sub_B99FD0(v48, v63, v62);
          }
          while ( v61 != v60 );
        }
      }
    }
    v42 = v48;
  }
  return sub_F162A0((__int64)a1, a3, v42);
}
