// Function: sub_342C970
// Address: 0x342c970
//
void __fastcall sub_342C970(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 *v3; // r12
  __int64 v4; // r13
  int v5; // r11d
  __int64 v6; // r8
  __int64 *v7; // rdx
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  int v11; // r15d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // r14
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // ecx
  int v22; // eax
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // r15
  unsigned __int8 *v29; // rsi
  _QWORD *v30; // r14
  _QWORD *v31; // rax
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 i; // rbx
  int v36; // eax
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // r15
  unsigned __int8 *v40; // rsi
  _QWORD *v41; // r14
  _QWORD *v42; // r15
  int v43; // eax
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // rdi
  __int64 *v47; // r8
  unsigned int v48; // ebx
  int v49; // r10d
  int v50; // r11d
  __int64 *v51; // r9
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 v55; // [rsp+20h] [rbp-B0h]
  __int64 *v57; // [rsp+38h] [rbp-98h]
  unsigned __int8 *v58; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int8 *v59; // [rsp+50h] [rbp-80h] BYREF
  __int64 v60; // [rsp+58h] [rbp-78h]
  __int64 v61; // [rsp+60h] [rbp-70h]
  __m128i v62; // [rsp+70h] [rbp-60h] BYREF
  __int64 v63; // [rsp+80h] [rbp-50h]
  __int64 v64; // [rsp+88h] [rbp-48h]
  int v65; // [rsp+90h] [rbp-40h]

  v2 = a2[11];
  if ( v2 )
  {
    v3 = (__int64 *)a2[41];
    v57 = a2 + 40;
    if ( a2 + 40 != v3 )
    {
      v4 = a2[11];
      v53 = v2 + 128;
      while ( 1 )
      {
        v17 = *(unsigned int *)(v4 + 152);
        v18 = v3[2];
        if ( (_DWORD)v17 )
        {
          v5 = 1;
          v6 = *(_QWORD *)(v4 + 136);
          v7 = 0;
          v8 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v9 = (__int64 *)(v6 + 16LL * v8);
          v10 = *v9;
          if ( v18 == *v9 )
          {
LABEL_5:
            v11 = *((_DWORD *)v9 + 2);
            goto LABEL_6;
          }
          while ( v10 != -4096 )
          {
            if ( v10 == -8192 && !v7 )
              v7 = v9;
            v8 = (v17 - 1) & (v5 + v8);
            v9 = (__int64 *)(v6 + 16LL * v8);
            v10 = *v9;
            if ( v18 == *v9 )
              goto LABEL_5;
            ++v5;
          }
          if ( !v7 )
            v7 = v9;
          v43 = *(_DWORD *)(v4 + 144);
          ++*(_QWORD *)(v4 + 128);
          v22 = v43 + 1;
          if ( 4 * v22 < (unsigned int)(3 * v17) )
          {
            if ( (int)v17 - *(_DWORD *)(v4 + 148) - v22 <= (unsigned int)v17 >> 3 )
            {
              sub_FF1B10(v53, v17);
              v44 = *(_DWORD *)(v4 + 152);
              if ( !v44 )
              {
LABEL_95:
                ++*(_DWORD *)(v4 + 144);
                BUG();
              }
              v45 = v44 - 1;
              v46 = *(_QWORD *)(v4 + 136);
              v47 = 0;
              v48 = v45 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v49 = 1;
              v22 = *(_DWORD *)(v4 + 144) + 1;
              v7 = (__int64 *)(v46 + 16LL * v48);
              v17 = *v7;
              if ( v18 != *v7 )
              {
                while ( v17 != -4096 )
                {
                  if ( !v47 && v17 == -8192 )
                    v47 = v7;
                  v48 = v45 & (v49 + v48);
                  v7 = (__int64 *)(v46 + 16LL * v48);
                  v17 = *v7;
                  if ( v18 == *v7 )
                    goto LABEL_16;
                  ++v49;
                }
                if ( v47 )
                  v7 = v47;
              }
            }
            goto LABEL_16;
          }
        }
        else
        {
          ++*(_QWORD *)(v4 + 128);
        }
        sub_FF1B10(v53, 2 * v17);
        v19 = *(_DWORD *)(v4 + 152);
        if ( !v19 )
          goto LABEL_95;
        v17 = (unsigned int)(v19 - 1);
        v20 = *(_QWORD *)(v4 + 136);
        v21 = v17 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v22 = *(_DWORD *)(v4 + 144) + 1;
        v7 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v7;
        if ( v18 != *v7 )
        {
          v50 = 1;
          v51 = 0;
          while ( v23 != -4096 )
          {
            if ( !v51 && v23 == -8192 )
              v51 = v7;
            v21 = v17 & (v50 + v21);
            v7 = (__int64 *)(v20 + 16LL * v21);
            v23 = *v7;
            if ( v18 == *v7 )
              goto LABEL_16;
            ++v50;
          }
          if ( v51 )
            v7 = v51;
        }
LABEL_16:
        *(_DWORD *)(v4 + 144) = v22;
        if ( *v7 != -4096 )
          --*(_DWORD *)(v4 + 148);
        *v7 = v18;
        v11 = 0;
        *((_DWORD *)v7 + 2) = 0;
LABEL_6:
        if ( !sub_AA4F90(v18) )
          goto LABEL_11;
        v14 = sub_2E311E0((__int64)v3);
        if ( (__int64 *)v14 == v3 + 6 )
          goto LABEL_11;
        v16 = *(_DWORD *)(v14 + 44);
        if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) & 0x200LL) != 0 )
            goto LABEL_11;
        }
        else
        {
          v17 = 512;
          if ( sub_2E88A90(v14, 512, 1) )
            goto LABEL_11;
        }
        v55 = sub_E6C430(a2[3], v17, v12, v13, v15);
        v52 = sub_E6C430(a2[3], v17, v24, v25, v26);
        sub_3017D20(v4, v11, v55, v52);
        v27 = **(_QWORD **)(a1 + 72);
        v28 = *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 160LL;
        if ( v27 )
        {
          v29 = *(unsigned __int8 **)(v27 + 48);
          v58 = v29;
          if ( v29 )
          {
            sub_B96E90((__int64)&v58, (__int64)v29, 1);
            v59 = v58;
            if ( v58 )
            {
              sub_B976B0((__int64)&v58, v58, (__int64)&v59);
              v58 = 0;
              v60 = 0;
              v61 = 0;
              v30 = (_QWORD *)v3[4];
              v62.m128i_i64[0] = (__int64)v59;
              if ( v59 )
                sub_B96E90((__int64)&v62, (__int64)v59, 1);
              v31 = sub_2E7B380(v30, v28, (unsigned __int8 **)&v62, 0);
              goto LABEL_26;
            }
          }
          else
          {
            v59 = 0;
          }
        }
        else
        {
          v58 = 0;
          v59 = 0;
        }
        v60 = 0;
        v61 = 0;
        v30 = (_QWORD *)v3[4];
        v62.m128i_i64[0] = 0;
        v31 = sub_2E7B380(v30, v28, (unsigned __int8 **)&v62, 0);
LABEL_26:
        v32 = (__int64)v31;
        if ( v62.m128i_i64[0] )
          sub_B91220((__int64)&v62, v62.m128i_i64[0]);
        sub_2E31040(v3 + 5, v32);
        v33 = *(_QWORD *)v14;
        v34 = *(_QWORD *)v32;
        *(_QWORD *)(v32 + 8) = v14;
        v33 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v32 = v33 | v34 & 7;
        *(_QWORD *)(v33 + 8) = v32;
        *(_QWORD *)v14 = v32 | *(_QWORD *)v14 & 7LL;
        if ( v60 )
          sub_2E882B0(v32, (__int64)v30, v60);
        if ( v61 )
          sub_2E88680(v32, (__int64)v30, v61);
        v62.m128i_i8[0] = 15;
        v63 = 0;
        v62.m128i_i32[0] &= 0xFFF000FF;
        v64 = v55;
        v62.m128i_i32[2] = 0;
        v65 = 0;
        sub_2E8EAD0(v32, (__int64)v30, &v62);
        if ( v59 )
          sub_B91220((__int64)&v59, (__int64)v59);
        if ( v58 )
          sub_B91220((__int64)&v58, (__int64)v58);
        for ( i = v3[6] & 0xFFFFFFFFFFFFFFF8LL; ; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        {
          v36 = *(_DWORD *)(i + 44);
          if ( (v36 & 4) != 0 || (v36 & 8) == 0 )
            break;
          if ( !sub_2E88A90(i, 512, 1) )
            goto LABEL_42;
LABEL_39:
          ;
        }
        if ( (*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) & 0x200LL) != 0 )
          goto LABEL_39;
LABEL_42:
        v37 = *(__int64 **)(i + 8);
        v38 = **(_QWORD **)(a1 + 72);
        v39 = *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 160LL;
        if ( !v38 )
        {
          v58 = 0;
          v59 = 0;
LABEL_77:
          v60 = 0;
          v61 = 0;
          v41 = (_QWORD *)v3[4];
          v62.m128i_i64[0] = 0;
          goto LABEL_47;
        }
        v40 = *(unsigned __int8 **)(v38 + 48);
        v58 = v40;
        if ( !v40 )
        {
          v59 = 0;
          goto LABEL_77;
        }
        sub_B96E90((__int64)&v58, (__int64)v40, 1);
        v59 = v58;
        if ( !v58 )
          goto LABEL_77;
        sub_B976B0((__int64)&v58, v58, (__int64)&v59);
        v58 = 0;
        v60 = 0;
        v61 = 0;
        v41 = (_QWORD *)v3[4];
        v62.m128i_i64[0] = (__int64)v59;
        if ( v59 )
          sub_B96E90((__int64)&v62, (__int64)v59, 1);
LABEL_47:
        v42 = sub_2E7B380(v41, v39, (unsigned __int8 **)&v62, 0);
        if ( v62.m128i_i64[0] )
          sub_B91220((__int64)&v62, v62.m128i_i64[0]);
        sub_2E326B0((__int64)v3, v37, (__int64)v42);
        if ( v60 )
          sub_2E882B0((__int64)v42, (__int64)v41, v60);
        if ( v61 )
          sub_2E88680((__int64)v42, (__int64)v41, v61);
        v62.m128i_i8[0] = 15;
        v63 = 0;
        v62.m128i_i32[0] &= 0xFFF000FF;
        v64 = v52;
        v62.m128i_i32[2] = 0;
        v65 = 0;
        sub_2E8EAD0((__int64)v42, (__int64)v41, &v62);
        if ( v59 )
          sub_B91220((__int64)&v59, (__int64)v59);
        if ( v58 )
        {
          sub_B91220((__int64)&v58, (__int64)v58);
          v3 = (__int64 *)v3[1];
          if ( v57 == v3 )
            return;
        }
        else
        {
LABEL_11:
          v3 = (__int64 *)v3[1];
          if ( v57 == v3 )
            return;
        }
      }
    }
  }
}
