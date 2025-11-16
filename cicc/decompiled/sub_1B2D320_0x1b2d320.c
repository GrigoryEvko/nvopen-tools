// Function: sub_1B2D320
// Address: 0x1b2d320
//
__int64 __fastcall sub_1B2D320(__int64 a1, int *a2, __int64 *a3, unsigned __int64 *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // r8d
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // rbx
  __int64 v22; // r8
  unsigned int v23; // edx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  unsigned __int64 *v27; // rdx
  __int64 v28; // rax
  int v29; // r14d
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  int v36; // r8d
  __int64 v37; // r14
  int v38; // edx
  char v39; // al
  __int64 v40; // rdx
  _QWORD *v41; // rcx
  __int64 v42; // rax
  unsigned int v43; // esi
  __int64 v44; // rdi
  unsigned int v45; // r14d
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // rdx
  int v49; // r11d
  __int64 *v50; // r10
  int v51; // eax
  int v52; // eax
  __int64 *v54; // r11
  int v55; // eax
  int v56; // ecx
  int v57; // eax
  int v58; // edi
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 *v62; // r8
  int v63; // eax
  int v64; // eax
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rdi
  int v68; // ecx
  int v69; // ecx
  __int64 v70; // rdi
  __int64 v71; // rdx
  __int64 v72; // rsi
  int v73; // r11d
  __int64 *v74; // r8
  int v75; // edx
  int v76; // edx
  __int64 v77; // rsi
  __int64 *v78; // rdi
  __int64 v79; // r14
  int v80; // r9d
  __int64 v81; // rcx
  __int64 v82; // [rsp+8h] [rbp-128h]
  int v83; // [rsp+8h] [rbp-128h]
  unsigned int v84; // [rsp+8h] [rbp-128h]
  __int64 v88; // [rsp+28h] [rbp-108h]
  unsigned __int64 *v89; // [rsp+38h] [rbp-F8h] BYREF
  _QWORD v90[2]; // [rsp+40h] [rbp-F0h] BYREF
  __m128i v91; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v92; // [rsp+60h] [rbp-D0h]
  _QWORD *v93; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v94; // [rsp+80h] [rbp-B0h]
  __m128i v95; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v96; // [rsp+A0h] [rbp-90h]
  __int64 v97; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-78h]
  __int64 v99; // [rsp+C0h] [rbp-70h]
  __int64 v100; // [rsp+C8h] [rbp-68h]
  __int64 v101; // [rsp+D0h] [rbp-60h]
  int v102; // [rsp+D8h] [rbp-58h]
  __int64 v103; // [rsp+E0h] [rbp-50h]
  __int64 v104; // [rsp+E8h] [rbp-48h]

  v5 = *a3;
  v6 = *((unsigned int *)a3 + 2);
  v7 = 48 * v6;
  v8 = v5 + 48 * v6;
  v9 = 48 * v6;
  if ( v5 != v8 )
  {
    v10 = v5 + 48 * v6;
    do
    {
      if ( *(_QWORD *)(v10 - 32) )
        break;
      v10 -= 48;
    }
    while ( v5 != v10 );
    v11 = v5 + v7 - (v8 - v10);
    if ( v11 != v8 )
    {
      do
      {
        v27 = a4;
        if ( v11 != v5 )
          v27 = *(unsigned __int64 **)(v11 - 32);
        v89 = v27;
        v28 = *(_QWORD *)(v11 + 32);
        v29 = *(_DWORD *)(v28 + 24);
        v88 = v28;
        if ( (v29 & 0xFFFFFFFD) != 0 )
        {
          if ( v29 != 1 )
            BUG();
          v12 = *(_QWORD *)(v28 + 48);
          v13 = sub_16498A0(v12);
          v97 = 0;
          v100 = v13;
          v101 = 0;
          v102 = 0;
          v103 = 0;
          v104 = 0;
          v98 = *(_QWORD *)(v12 + 40);
          v99 = v12 + 24;
          v14 = *(_QWORD *)(v12 + 48);
          v95.m128i_i64[0] = v14;
          if ( v14 )
          {
            sub_1623A60((__int64)&v95, v14, 2);
            if ( v97 )
              sub_161E7C0((__int64)&v97, v97);
            v97 = v95.m128i_i64[0];
            if ( v95.m128i_i64[0] )
              sub_1623210((__int64)&v95, (unsigned __int8 *)v95.m128i_i64[0], (__int64)&v97);
          }
          v15 = sub_1B2A270(*(__int64 **)(*(_QWORD *)(a1 + 16) + 40LL), *v89);
          v18 = v15;
          if ( !*(_QWORD *)(v15 + 8) )
          {
            v95.m128i_i64[0] = v15;
            v82 = v15;
            sub_1B2BF50(a1 + 3264, (unsigned __int64 *)&v95, v15, v16, v17);
            v18 = v82;
          }
          LOWORD(v96) = 257;
          v19 = sub_1285290(&v97, *(_QWORD *)(v18 + 24), v18, (int)&v89, 1, (__int64)&v95, 0);
          v20 = *(_DWORD *)(a1 + 104);
          v21 = v19;
          if ( !v20 )
          {
            ++*(_QWORD *)(a1 + 80);
            goto LABEL_59;
          }
          v22 = *(_QWORD *)(a1 + 88);
          v23 = ((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4);
          v24 = (v20 - 1) & v23;
          v25 = (__int64 *)(v22 + 16LL * v24);
          v26 = *v25;
          if ( v21 != *v25 )
          {
            v83 = 1;
            v54 = 0;
            while ( v26 != -8 )
            {
              if ( v54 || v26 != -16 )
                v25 = v54;
              v24 = (v20 - 1) & (v83 + v24);
              v26 = *(_QWORD *)(v22 + 16LL * v24);
              if ( v21 == v26 )
                goto LABEL_17;
              ++v83;
              v54 = v25;
              v25 = (__int64 *)(v22 + 16LL * v24);
            }
            if ( !v54 )
              v54 = v25;
            v55 = *(_DWORD *)(a1 + 96);
            ++*(_QWORD *)(a1 + 80);
            v56 = v55 + 1;
            if ( 4 * (v55 + 1) >= 3 * v20 )
            {
LABEL_59:
              sub_1B2D160(a1 + 80, 2 * v20);
              v57 = *(_DWORD *)(a1 + 104);
              if ( !v57 )
                goto LABEL_116;
              v58 = v57 - 1;
              v59 = *(_QWORD *)(a1 + 88);
              LODWORD(v60) = (v57 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
              v56 = *(_DWORD *)(a1 + 96) + 1;
              v54 = (__int64 *)(v59 + 16LL * (unsigned int)v60);
              v61 = *v54;
              if ( v21 != *v54 )
              {
                v62 = 0;
                while ( v61 != -8 )
                {
                  if ( !v62 && v61 == -16 )
                    v62 = v54;
                  v60 = v58 & (unsigned int)(v60 + v29);
                  v54 = (__int64 *)(v59 + 16 * v60);
                  v61 = *v54;
                  if ( v21 == *v54 )
                    goto LABEL_54;
                  ++v29;
                }
LABEL_63:
                if ( v62 )
                  v54 = v62;
              }
            }
            else if ( v20 - *(_DWORD *)(a1 + 100) - v56 <= v20 >> 3 )
            {
              v84 = v23;
              sub_1B2D160(a1 + 80, v20);
              v63 = *(_DWORD *)(a1 + 104);
              if ( !v63 )
              {
LABEL_116:
                ++*(_DWORD *)(a1 + 96);
                BUG();
              }
              v64 = v63 - 1;
              v65 = *(_QWORD *)(a1 + 88);
              v62 = 0;
              LODWORD(v66) = v64 & v84;
              v54 = (__int64 *)(v65 + 16LL * (v64 & v84));
              v67 = *v54;
              v56 = *(_DWORD *)(a1 + 96) + 1;
              if ( v21 != *v54 )
              {
                while ( v67 != -8 )
                {
                  if ( v67 == -16 && !v62 )
                    v62 = v54;
                  v66 = v64 & (unsigned int)(v66 + v29);
                  v54 = (__int64 *)(v65 + 16 * v66);
                  v67 = *v54;
                  if ( v21 == *v54 )
                    goto LABEL_54;
                  ++v29;
                }
                goto LABEL_63;
              }
            }
LABEL_54:
            *(_DWORD *)(a1 + 96) = v56;
            if ( *v54 != -8 )
              --*(_DWORD *)(a1 + 100);
            *v54 = v21;
            v54[1] = v88;
          }
        }
        else
        {
          v30 = sub_157EBA0(*(_QWORD *)(v28 + 48));
          v31 = sub_16498A0(v30);
          v97 = 0;
          v100 = v31;
          v101 = 0;
          v102 = 0;
          v103 = 0;
          v104 = 0;
          v98 = *(_QWORD *)(v30 + 40);
          v99 = v30 + 24;
          v32 = *(_QWORD *)(v30 + 48);
          v95.m128i_i64[0] = v32;
          if ( v32 )
          {
            sub_1623A60((__int64)&v95, v32, 2);
            if ( v97 )
              sub_161E7C0((__int64)&v97, v97);
            v97 = v95.m128i_i64[0];
            if ( v95.m128i_i64[0] )
              sub_1623210((__int64)&v95, (unsigned __int8 *)v95.m128i_i64[0], (__int64)&v97);
          }
          v33 = sub_1B2A270(*(__int64 **)(*(_QWORD *)(a1 + 16) + 40LL), *v89);
          v37 = v33;
          if ( !*(_QWORD *)(v33 + 8) )
          {
            v95.m128i_i64[0] = v33;
            sub_1B2BF50(a1 + 3264, (unsigned __int64 *)&v95, v34, v35, v36);
          }
          v94 = 265;
          v38 = *a2 + 1;
          LODWORD(v93) = *a2;
          *a2 = v38;
          v90[0] = sub_1649960((__int64)v89);
          LOWORD(v92) = 773;
          v91.m128i_i64[0] = (__int64)v90;
          v91.m128i_i64[1] = (__int64)".";
          v39 = v94;
          v90[1] = v40;
          if ( (_BYTE)v94 )
          {
            if ( (_BYTE)v94 == 1 )
            {
              v95 = _mm_loadu_si128(&v91);
              v96 = v92;
            }
            else
            {
              v41 = v93;
              if ( HIBYTE(v94) != 1 )
              {
                v41 = &v93;
                v39 = 2;
              }
              v95.m128i_i64[1] = (__int64)v41;
              v95.m128i_i64[0] = (__int64)&v91;
              LOBYTE(v96) = 2;
              BYTE1(v96) = v39;
            }
          }
          else
          {
            LOWORD(v96) = 256;
          }
          v42 = sub_1285290(&v97, *(_QWORD *)(v37 + 24), v37, (int)&v89, 1, (__int64)&v95, 0);
          v43 = *(_DWORD *)(a1 + 104);
          v21 = v42;
          if ( !v43 )
          {
            ++*(_QWORD *)(a1 + 80);
            goto LABEL_75;
          }
          v44 = *(_QWORD *)(a1 + 88);
          v45 = ((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4);
          v46 = (v43 - 1) & v45;
          v47 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v47;
          if ( v21 != *v47 )
          {
            v49 = 1;
            v50 = 0;
            while ( v48 != -8 )
            {
              if ( v48 != -16 || v50 )
                v47 = v50;
              v46 = (v43 - 1) & (v49 + v46);
              v48 = *(_QWORD *)(v44 + 16LL * v46);
              if ( v21 == v48 )
                goto LABEL_17;
              ++v49;
              v50 = v47;
              v47 = (__int64 *)(v44 + 16LL * v46);
            }
            if ( !v50 )
              v50 = v47;
            v51 = *(_DWORD *)(a1 + 96);
            ++*(_QWORD *)(a1 + 80);
            v52 = v51 + 1;
            if ( 4 * v52 >= 3 * v43 )
            {
LABEL_75:
              sub_1B2D160(a1 + 80, 2 * v43);
              v68 = *(_DWORD *)(a1 + 104);
              if ( !v68 )
                goto LABEL_117;
              v69 = v68 - 1;
              v70 = *(_QWORD *)(a1 + 88);
              LODWORD(v71) = v69 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
              v52 = *(_DWORD *)(a1 + 96) + 1;
              v50 = (__int64 *)(v70 + 16LL * (unsigned int)v71);
              v72 = *v50;
              if ( v21 != *v50 )
              {
                v73 = 1;
                v74 = 0;
                while ( v72 != -8 )
                {
                  if ( !v74 && v72 == -16 )
                    v74 = v50;
                  v71 = v69 & (unsigned int)(v71 + v73);
                  v50 = (__int64 *)(v70 + 16 * v71);
                  v72 = *v50;
                  if ( v21 == *v50 )
                    goto LABEL_44;
                  ++v73;
                }
                if ( v74 )
                  v50 = v74;
              }
            }
            else if ( v43 - *(_DWORD *)(a1 + 100) - v52 <= v43 >> 3 )
            {
              sub_1B2D160(a1 + 80, v43);
              v75 = *(_DWORD *)(a1 + 104);
              if ( !v75 )
              {
LABEL_117:
                ++*(_DWORD *)(a1 + 96);
                BUG();
              }
              v76 = v75 - 1;
              v77 = *(_QWORD *)(a1 + 88);
              v78 = 0;
              LODWORD(v79) = v76 & v45;
              v80 = 1;
              v52 = *(_DWORD *)(a1 + 96) + 1;
              v50 = (__int64 *)(v77 + 16LL * (unsigned int)v79);
              v81 = *v50;
              if ( v21 != *v50 )
              {
                while ( v81 != -8 )
                {
                  if ( v81 == -16 && !v78 )
                    v78 = v50;
                  v79 = v76 & (unsigned int)(v79 + v80);
                  v50 = (__int64 *)(v77 + 16 * v79);
                  v81 = *v50;
                  if ( v21 == *v50 )
                    goto LABEL_44;
                  ++v80;
                }
                if ( v78 )
                  v50 = v78;
              }
            }
LABEL_44:
            *(_DWORD *)(a1 + 96) = v52;
            if ( *v50 != -8 )
              --*(_DWORD *)(a1 + 100);
            *v50 = v21;
            v50[1] = v88;
          }
        }
LABEL_17:
        *(_QWORD *)(v11 + 16) = v21;
        if ( v97 )
          sub_161E7C0((__int64)&v97, v97);
        v11 += 48;
        v5 = *a3;
        v9 = 48LL * *((unsigned int *)a3 + 2);
      }
      while ( v11 != *a3 + v9 );
    }
  }
  return *(_QWORD *)(v5 + v9 - 32);
}
