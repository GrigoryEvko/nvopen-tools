// Function: sub_205E850
// Address: 0x205e850
//
void __fastcall sub_205E850(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // rdx
  char v7; // di
  int v8; // ebx
  int v9; // r8d
  int v10; // r9d
  size_t v11; // r12
  _BYTE *v12; // rdi
  signed __int64 v13; // rsi
  signed __int64 v14; // rax
  signed __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r14
  unsigned __int64 v18; // r12
  __int64 *v19; // r13
  unsigned __int32 v20; // r13d
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // r13
  unsigned int v23; // r12d
  unsigned __int64 v24; // r13
  void *v25; // r14
  signed __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // r14d
  unsigned int v30; // ebx
  signed __int64 v31; // r12
  __int64 v32; // rax
  unsigned int v33; // ecx
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rcx
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  int v43; // r8d
  int v44; // r9d
  __int64 v45; // r15
  unsigned int v46; // r13d
  __int64 v47; // rdi
  unsigned int v48; // edx
  __int64 v49; // rax
  __int64 v50; // [rsp+8h] [rbp-148h]
  signed __int64 v52; // [rsp+28h] [rbp-128h]
  signed __int64 v53; // [rsp+30h] [rbp-120h]
  __int64 v54; // [rsp+38h] [rbp-118h]
  __int64 v56; // [rsp+58h] [rbp-F8h]
  __int64 v58; // [rsp+68h] [rbp-E8h]
  signed __int64 v59; // [rsp+78h] [rbp-D8h]
  unsigned int v60; // [rsp+78h] [rbp-D8h]
  __int64 v61; // [rsp+80h] [rbp-D0h] BYREF
  unsigned __int32 v62; // [rsp+88h] [rbp-C8h]
  __m128i v63; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v64; // [rsp+A0h] [rbp-B0h] BYREF
  int v65; // [rsp+B0h] [rbp-A0h]
  void *v66; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v67; // [rsp+C8h] [rbp-88h]
  _BYTE v68[32]; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE *v69; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v70; // [rsp+F8h] [rbp-58h]
  _BYTE v71[80]; // [rsp+100h] [rbp-50h] BYREF

  if ( (unsigned int)sub_1700720(a1[68]) )
  {
    v3 = a1[69];
    v4 = *(_QWORD *)(v3 + 16);
    v56 = sub_1E0A0C0(*(_QWORD *)(v3 + 32));
    v5 = 8 * sub_15A9520(v56, 0);
    if ( v5 == 32 )
    {
      v6 = 5;
      v7 = 5;
    }
    else if ( v5 > 0x20 )
    {
      if ( v5 == 64 )
      {
        v6 = 6;
        v7 = 6;
      }
      else
      {
        if ( v5 != 128 )
          return;
        v6 = 7;
        v7 = 7;
      }
    }
    else if ( v5 == 8 )
    {
      v6 = 3;
      v7 = 3;
    }
    else
    {
      v6 = 4;
      v7 = 4;
      if ( v5 != 16 )
        return;
    }
    if ( *(_QWORD *)(v4 + 8 * v6 + 120) && !*(_BYTE *)(v4 + 259LL * (v7 & 7) + 2544) )
    {
      v8 = sub_2045180(v7);
      v67 = 0x800000000LL;
      v50 = a2[1] - *a2;
      v53 = 0xCCCCCCCCCCCCCCCDLL * (v50 >> 3);
      v66 = v68;
      v11 = -3435973836LL * (unsigned int)(v50 >> 3);
      if ( (unsigned __int64)v50 > 0x140 )
      {
        sub_16CD150((__int64)&v66, v68, 0xCCCCCCCCCCCCCCCDLL * (v50 >> 3), 4, v9, v10);
        LODWORD(v67) = -858993459 * (v50 >> 3);
        if ( v66 != (char *)v66 + v11 )
          memset(v66, 0, v11);
        v69 = v71;
        v70 = 0x800000000LL;
        sub_16CD150((__int64)&v69, v71, v53, 4, v43, v44);
        v12 = v69;
      }
      else
      {
        LODWORD(v67) = -858993459 * (v50 >> 3);
        if ( v68 != &v68[v11] )
          memset(v68, 0, -3435973836LL * (unsigned int)(v50 >> 3));
        HIDWORD(v70) = 8;
        v69 = v71;
        v12 = v71;
      }
      LODWORD(v70) = -858993459 * (v50 >> 3);
      if ( v11 )
        memset(v12, 0, v11);
      v13 = v53 - 1;
      *((_DWORD *)v66 + v13) = 1;
      *(_DWORD *)&v69[4 * v13] = v53 - 1;
      v59 = v53 - 2;
      if ( v53 - 2 < 0 )
        goto LABEL_41;
      v14 = v59 + v8;
      v58 = v50 - 80;
      while ( 1 )
      {
        v54 = 4 * v59;
        *((_DWORD *)v66 + v59) = *((_DWORD *)v66 + v59 + 1) + 1;
        *(_DWORD *)&v69[4 * v59] = v59;
        v15 = v14 - 1;
        v52 = v14 - 1;
        if ( v53 <= v14 )
          v15 = v53 - 1;
        if ( v59 >= v15 )
          goto LABEL_40;
        while ( 1 )
        {
          v16 = *(_QWORD *)(*a2 + v58 + 8);
          v17 = *(_QWORD *)(*a2 + 40 * v15 + 16);
          v18 = 8 * (unsigned int)sub_15A95A0(v56, 0);
          v19 = (__int64 *)(v16 + 24);
          v63.m128i_i32[2] = *(_DWORD *)(v17 + 32);
          if ( v63.m128i_i32[2] > 0x40u )
            sub_16A4FD0((__int64)&v63, (const void **)(v17 + 24));
          else
            v63.m128i_i64[0] = *(_QWORD *)(v17 + 24);
          sub_16A7590((__int64)&v63, v19);
          v20 = v63.m128i_u32[2];
          v21 = (unsigned __int64 *)v63.m128i_i64[0];
          v63.m128i_i32[2] = 0;
          v62 = v20;
          v61 = v63.m128i_i64[0];
          if ( v20 > 0x40 )
            break;
          if ( v63.m128i_i64[0] != -1 )
          {
            v22 = v63.m128i_i64[0] + 1;
            goto LABEL_32;
          }
LABEL_22:
          if ( v59 == --v15 )
            goto LABEL_40;
        }
        if ( v20 - (unsigned int)sub_16A57B0((__int64)&v61) > 0x40 )
          break;
        v22 = *v21;
        if ( *v21 != -1 )
        {
          ++v22;
          goto LABEL_28;
        }
        j_j___libc_free_0_0(v21);
        if ( v63.m128i_i32[2] <= 0x40u )
          goto LABEL_22;
LABEL_30:
        if ( v63.m128i_i64[0] )
          j_j___libc_free_0_0(v63.m128i_i64[0]);
LABEL_32:
        if ( v22 > v18 )
          goto LABEL_22;
        v23 = ((unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1[89] + 8LL) + 104LL)
                                      - *(_QWORD *)(*(_QWORD *)(a1[89] + 8LL) + 96LL)) >> 3)
             + 63) >> 6;
        v24 = 8LL * v23;
        v25 = (void *)malloc(v24);
        if ( !v25 )
        {
          if ( v24 || (v49 = malloc(1u)) == 0 )
            sub_16BD1C0("Allocation failed", 1u);
          else
            v25 = (void *)v49;
        }
        if ( !v23 )
        {
          if ( v59 <= v15 )
            goto LABEL_36;
          goto LABEL_67;
        }
        memset(v25, 0, v24);
        if ( v59 > v15 )
        {
LABEL_64:
          v45 = 0;
          v46 = 0;
          do
          {
            v47 = *((_QWORD *)v25 + v45++);
            v46 += sub_39FAC40(v47);
          }
          while ( v23 > (unsigned int)v45 );
          if ( v46 > 3 )
            goto LABEL_39;
LABEL_67:
          v48 = 1;
          if ( v53 - 1 != v15 )
            v48 = *((_DWORD *)v66 + v15 + 1) + 1;
          if ( *(_DWORD *)((char *)v66 + v54) > v48 )
          {
            *(_DWORD *)((char *)v66 + v54) = v48;
            *(_DWORD *)&v69[4 * v59] = v15;
          }
          _libc_free((unsigned __int64)v25);
          goto LABEL_22;
        }
LABEL_36:
        v26 = v59;
        v27 = *a2 + v58;
        while ( !*(_DWORD *)v27 )
        {
          v28 = *(_QWORD *)(v27 + 24);
          ++v26;
          v27 += 40;
          *((_QWORD *)v25 + (*(_DWORD *)(v28 + 48) >> 6)) |= 1LL << *(_DWORD *)(v28 + 48);
          if ( v26 > v15 )
          {
            if ( !v23 )
              goto LABEL_67;
            goto LABEL_64;
          }
        }
LABEL_39:
        _libc_free((unsigned __int64)v25);
LABEL_40:
        --v59;
        v58 -= 40;
        v14 = v52;
        if ( v59 == -1 )
        {
LABEL_41:
          if ( v50 <= 0 )
          {
            v37 = 0;
            v38 = a2[1];
            v39 = *a2;
            v41 = 0xCCCCCCCCCCCCCCCDLL * ((v38 - *a2) >> 3);
            goto LABEL_47;
          }
          v29 = 0;
          v30 = 0;
          v31 = 0;
          do
          {
            while ( 1 )
            {
              v33 = *(_DWORD *)&v69[4 * v31];
              v65 = -1;
              v60 = v33;
              v34 = 5LL * v29;
              v35 = 40LL * v29;
              if ( !(unsigned __int8)sub_205E830(a1, a2, v30, v33, a3, (__int64)&v63) )
                break;
              v32 = *a2;
              v31 = v60 + 1;
              ++v29;
              v30 = v60 + 1;
              *(__m128i *)(v32 + 8 * v34) = _mm_loadu_si128(&v63);
              *(__m128i *)(v32 + v35 + 16) = _mm_loadu_si128(&v64);
              *(_DWORD *)(v32 + v35 + 32) = v65;
              if ( v31 >= v53 )
                goto LABEL_46;
            }
            v36 = 1 - v30 + v60;
            v29 += v36;
            memmove((void *)(*a2 + v35), (const void *)(*a2 + 40 * v31), 40 * v36);
            v31 = v60 + 1;
            v30 = v60 + 1;
          }
          while ( v31 < v53 );
LABEL_46:
          v37 = v29;
          v38 = a2[1];
          v39 = *a2;
          v40 = 0xCCCCCCCCCCCCCCCDLL * ((v38 - *a2) >> 3);
          v41 = v40;
          if ( v40 < v29 )
          {
            sub_205A2F0((const __m128i **)a2, v29 - v40);
          }
          else
          {
LABEL_47:
            if ( v37 < v41 )
            {
              v42 = v39 + 40 * v37;
              if ( v38 != v42 )
                a2[1] = v42;
            }
          }
          if ( v69 != v71 )
            _libc_free((unsigned __int64)v69);
          if ( v66 != v68 )
            _libc_free((unsigned __int64)v66);
          return;
        }
      }
      v22 = -1;
LABEL_28:
      if ( !v21 )
        goto LABEL_32;
      j_j___libc_free_0_0(v21);
      if ( v63.m128i_i32[2] <= 0x40u )
        goto LABEL_32;
      goto LABEL_30;
    }
  }
}
