// Function: sub_359E620
// Address: 0x359e620
//
__int64 __fastcall sub_359E620(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        signed int a10,
        char a11)
{
  __int64 result; // rax
  __int64 v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // rax
  int v16; // ebx
  __int64 v17; // r13
  unsigned int v18; // eax
  __int64 v19; // rsi
  unsigned int v20; // r12d
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // eax
  bool v26; // cl
  bool v27; // dl
  __int64 v28; // rdx
  __int64 v29; // rax
  int *v30; // rax
  unsigned __int64 v31; // rax
  unsigned int v32; // ebx
  int *i; // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r9
  __int32 v37; // eax
  __int64 v38; // r15
  __int64 v39; // rax
  _QWORD *v40; // r12
  __int64 v41; // r8
  _QWORD *v42; // rax
  __int64 *v43; // r8
  __int64 v44; // r15
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // r12
  unsigned int v48; // esi
  __int64 v49; // r8
  int v50; // r11d
  __int64 *v51; // rcx
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // rdi
  __int64 *v55; // rax
  int v56; // eax
  int v57; // edx
  __int64 v58; // [rsp+10h] [rbp-130h]
  __int64 *v60; // [rsp+28h] [rbp-118h]
  __int64 v61; // [rsp+38h] [rbp-108h]
  unsigned int v64; // [rsp+5Ch] [rbp-E4h]
  __int64 v65; // [rsp+60h] [rbp-E0h]
  unsigned int v67; // [rsp+70h] [rbp-D0h]
  unsigned int v68; // [rsp+74h] [rbp-CCh]
  __int64 v69; // [rsp+78h] [rbp-C8h]
  __int64 *v70; // [rsp+78h] [rbp-C8h]
  __int64 *v71; // [rsp+78h] [rbp-C8h]
  unsigned int v72; // [rsp+80h] [rbp-C0h]
  int v73; // [rsp+84h] [rbp-BCh]
  int v74; // [rsp+90h] [rbp-B0h]
  __int32 v75; // [rsp+94h] [rbp-ACh]
  __int64 v76; // [rsp+98h] [rbp-A8h]
  unsigned int v77; // [rsp+ACh] [rbp-94h] BYREF
  __int64 v78; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v80; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v81; // [rsp+C8h] [rbp-78h]
  __int64 v82; // [rsp+D0h] [rbp-70h]
  __m128i v83; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v84; // [rsp+F0h] [rbp-50h]
  __int64 v85; // [rsp+F8h] [rbp-48h]
  __int64 v86; // [rsp+100h] [rbp-40h]

  v72 = a10 - a9;
  if ( a10 == a9 )
  {
    v67 = a9 - 1;
    v64 = a10;
  }
  else
  {
    v67 = 2 * a9 - a10;
    v64 = a10 - 1;
  }
  result = sub_2E311E0(a1[6]);
  v13 = a1[6];
  v78 = result;
  v58 = v13 + 48;
  if ( v13 + 48 != result )
  {
    v60 = a1 + 11;
    v61 = a6 + 32LL * v67;
    while ( 1 )
    {
      v14 = *(_DWORD *)(result + 40) & 0xFFFFFF;
      if ( v14 )
        break;
LABEL_35:
      sub_2FD79B0(&v78);
      result = v78;
      if ( v58 == v78 )
        return result;
    }
    v76 = 0;
    v65 = 40LL * v14;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v78 + 32) + v76;
      if ( *(_BYTE *)v15 )
        goto LABEL_7;
      if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
        goto LABEL_7;
      v16 = *(_DWORD *)(v15 + 8);
      if ( v16 >= 0 )
        goto LABEL_7;
      v17 = *a1;
      v18 = sub_3598DB0(*a1, v78);
      LODWORD(v80) = v16;
      v19 = (__int64)v60;
      v20 = v18;
      v21 = a1[12];
      v77 = v16;
      if ( !v21 )
        goto LABEL_18;
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 + 16);
          v23 = *(_QWORD *)(v21 + 24);
          if ( (unsigned int)v16 <= *(_DWORD *)(v21 + 32) )
            break;
          v21 = *(_QWORD *)(v21 + 24);
          if ( !v23 )
            goto LABEL_16;
        }
        v19 = v21;
        v21 = *(_QWORD *)(v21 + 16);
      }
      while ( v22 );
LABEL_16:
      if ( v60 == (__int64 *)v19 || (unsigned int)v16 < *(_DWORD *)(v19 + 32) )
      {
LABEL_18:
        v83.m128i_i64[0] = (__int64)&v80;
        v24 = sub_359C130(a1 + 10, v19, (unsigned int **)&v83);
        v17 = *a1;
        v19 = v24;
      }
      v25 = *(_DWORD *)(v19 + 36);
      if ( *(_DWORD *)(v17 + 96) <= a10 )
      {
        if ( v25 )
          goto LABEL_87;
        if ( *(_BYTE *)(v19 + 40) )
        {
          v25 = 1;
LABEL_87:
          v27 = a10 != a9;
          goto LABEL_40;
        }
        v26 = 1;
        v27 = a10 != a9;
        if ( !v72 )
          goto LABEL_40;
      }
      else
      {
        v26 = v25 == 0;
        v27 = a10 != a9;
        if ( !v72 )
          goto LABEL_40;
      }
      if ( v26 )
      {
        if ( v20 )
          goto LABEL_7;
        v28 = a1[3];
        v29 = (v77 & 0x80000000) != 0
            ? *(_QWORD *)(*(_QWORD *)(v28 + 56) + 16LL * (v77 & 0x7FFFFFFF) + 8)
            : *(_QWORD *)(*(_QWORD *)(v28 + 304) + 8LL * v77);
        if ( !v29 )
          goto LABEL_7;
        while ( (*(_BYTE *)(v29 + 3) & 0x10) != 0 )
        {
          v29 = *(_QWORD *)(v29 + 32);
          if ( !v29 )
            goto LABEL_7;
        }
LABEL_27:
        if ( a1[6] == *(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL) )
        {
          while ( 1 )
          {
            v29 = *(_QWORD *)(v29 + 32);
            if ( !v29 )
              goto LABEL_7;
            if ( (*(_BYTE *)(v29 + 3) & 0x10) == 0 )
              goto LABEL_27;
          }
        }
        v73 = 0;
        v68 = v67 != -1;
        goto LABEL_50;
      }
LABEL_40:
      if ( v20 > v67 && v27 )
        goto LABEL_7;
      v73 = 0;
      if ( v67 + 1 - v20 <= v25 )
        v25 = v67 + 1 - v20;
      v68 = v25;
      if ( !v72 )
      {
        v30 = sub_2FFAE70(a6 + 32LL * v64, (int *)&v77);
        v73 = *v30;
        v31 = sub_2EBEE10(a1[3], *v30);
        if ( v31 )
        {
          if ( (!*(_WORD *)(v31 + 68) || *(_WORD *)(v31 + 68) == 68) && a2 == *(_QWORD *)(v31 + 24) )
            v73 = sub_3598190(v31, a4);
        }
      }
LABEL_50:
      if ( v68 )
      {
        v32 = 0;
        for ( i = sub_2FFAE70(v61, (int *)&v77); ; i = sub_2FFAE70(v61, (int *)&v77) )
        {
          v74 = *i;
          if ( v67 >= v32 )
            v74 = *sub_2FFAE70(a6 + 32LL * (v67 - v32), (int *)&v77);
          if ( v72 )
          {
            if ( v64 != a9 || v32 )
              v73 = *sub_2FFAE70(a7 + 32LL * (v64 - v32), (int *)&v77);
            else
              v73 = *sub_2FFAE70(a6 + 32LL * a9, (int *)&v77);
          }
          v37 = sub_2EC06C0(
                  a1[3],
                  *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v77 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  byte_3F871B3,
                  0,
                  v34,
                  v35);
          v79 = 0;
          v75 = v37;
          v38 = *(_QWORD *)(a1[4] + 8);
          v80 = 0;
          v81 = 0;
          v82 = 0;
          v39 = sub_2E311E0(a2);
          v40 = *(_QWORD **)(a2 + 32);
          v41 = v39;
          v83.m128i_i64[0] = v80;
          if ( v80 )
          {
            v69 = v39;
            sub_B96E90((__int64)&v83, v80, 1);
            v41 = v69;
          }
          v70 = (__int64 *)v41;
          v42 = sub_2E7B380(v40, v38, (unsigned __int8 **)&v83, 0);
          v43 = v70;
          v44 = (__int64)v42;
          if ( v83.m128i_i64[0] )
          {
            sub_B91220((__int64)&v83, v83.m128i_i64[0]);
            v43 = v70;
          }
          v71 = v43;
          sub_2E31040((__int64 *)(a2 + 40), v44);
          v45 = *v71;
          v46 = *(_QWORD *)v44 & 7LL;
          *(_QWORD *)(v44 + 8) = v71;
          v45 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v44 = v45 | v46;
          *(_QWORD *)(v45 + 8) = v44;
          *v71 = v44 | *v71 & 7;
          if ( v81 )
            sub_2E882B0(v44, (__int64)v40, v81);
          if ( v82 )
            sub_2E88680(v44, (__int64)v40, v82);
          v83.m128i_i64[0] = 0x10000000;
          v84 = 0;
          v83.m128i_i32[2] = v75;
          v85 = 0;
          v86 = 0;
          sub_2E8EAD0(v44, (__int64)v40, &v83);
          if ( v80 )
            sub_B91220((__int64)&v80, v80);
          if ( v79 )
            sub_B91220((__int64)&v79, v79);
          v83.m128i_i64[0] = 0;
          v83.m128i_i32[2] = v74;
          v84 = 0;
          v85 = 0;
          v86 = 0;
          sub_2E8EAD0(v44, (__int64)v40, &v83);
          v83.m128i_i8[0] = 4;
          v84 = 0;
          v83.m128i_i32[0] &= 0xFFF000FF;
          v85 = a3;
          sub_2E8EAD0(v44, (__int64)v40, &v83);
          v83.m128i_i64[0] = 0;
          v84 = 0;
          v83.m128i_i32[2] = v73;
          v85 = 0;
          v86 = 0;
          sub_2E8EAD0(v44, (__int64)v40, &v83);
          v83.m128i_i8[0] = 4;
          v84 = 0;
          v83.m128i_i32[0] &= 0xFFF000FF;
          v85 = a4;
          sub_2E8EAD0(v44, (__int64)v40, &v83);
          if ( !v32 )
            break;
LABEL_75:
          if ( v72 )
          {
            *sub_2FFAE70(a7 + 32LL * (a10 - v32), (int *)&v77) = v75;
            if ( v68 - 1 == v32 )
            {
              sub_3599870(a1, a2, a8, a10, v32, v78, v77, v75, 0);
              if ( a11 )
                goto LABEL_78;
            }
          }
          else
          {
            sub_3599870(a1, a2, a8, a10, v32, v78, v74, v75, 0);
            sub_3599870(a1, a2, a8, a10, v32, v78, v73, v75, 0);
            *sub_2FFAE70(a7 + 32LL * (v64 - 1 - v32), (int *)&v77) = v75;
            if ( !a11 )
            {
              v73 = v75;
              goto LABEL_54;
            }
            v73 = v75;
            if ( v68 - 1 == v32 )
LABEL_78:
              sub_35988E0(v77, v75, a1[6], a1[3], a1[5], v36);
          }
LABEL_54:
          if ( v68 == ++v32 )
            goto LABEL_7;
        }
        v47 = v78;
        v80 = v44;
        v48 = *(_DWORD *)(a8 + 24);
        if ( v48 )
        {
          v49 = *(_QWORD *)(a8 + 8);
          v50 = 1;
          v51 = 0;
          v52 = (v48 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v53 = (__int64 *)(v49 + 16LL * v52);
          v54 = *v53;
          if ( v44 == *v53 )
          {
LABEL_84:
            v55 = v53 + 1;
LABEL_85:
            *v55 = v47;
            goto LABEL_75;
          }
          while ( v54 != -4096 )
          {
            if ( v54 == -8192 && !v51 )
              v51 = v53;
            v52 = (v48 - 1) & (v50 + v52);
            v53 = (__int64 *)(v49 + 16LL * v52);
            v54 = *v53;
            if ( v44 == *v53 )
              goto LABEL_84;
            ++v50;
          }
          if ( !v51 )
            v51 = v53;
          ++*(_QWORD *)a8;
          v56 = *(_DWORD *)(a8 + 16);
          v83.m128i_i64[0] = (__int64)v51;
          v57 = v56 + 1;
          if ( 4 * (v56 + 1) < 3 * v48 )
          {
            if ( v48 - *(_DWORD *)(a8 + 20) - v57 > v48 >> 3 )
            {
LABEL_99:
              *(_DWORD *)(a8 + 16) = v57;
              if ( *v51 != -4096 )
                --*(_DWORD *)(a8 + 20);
              *v51 = v44;
              v55 = v51 + 1;
              v51[1] = 0;
              goto LABEL_85;
            }
LABEL_104:
            sub_2E48800(a8, v48);
            sub_3547B30(a8, &v80, &v83);
            v44 = v80;
            v51 = (__int64 *)v83.m128i_i64[0];
            v57 = *(_DWORD *)(a8 + 16) + 1;
            goto LABEL_99;
          }
        }
        else
        {
          v83.m128i_i64[0] = 0;
          ++*(_QWORD *)a8;
        }
        v48 *= 2;
        goto LABEL_104;
      }
LABEL_7:
      v76 += 40;
      if ( v65 == v76 )
        goto LABEL_35;
    }
  }
  return result;
}
