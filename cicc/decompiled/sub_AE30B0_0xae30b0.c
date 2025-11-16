// Function: sub_AE30B0
// Address: 0xae30b0
//
unsigned __int64 *__fastcall sub_AE30B0(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r11
  __int64 v7; // r15
  __int8 *v8; // r13
  char *v9; // rax
  __int8 v10; // al
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned __int64 v18; // r15
  __int64 v19; // rbx
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  const char *v27; // rbx
  unsigned int v28; // ebx
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // r11
  __m128i v32; // xmm6
  __m128i v33; // xmm7
  __int64 v34; // r14
  unsigned __int64 v35; // r8
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // r13
  char *v39; // rdx
  unsigned int v40; // ebx
  char *v41; // rsi
  __int64 v42; // rax
  char v43; // bl
  __int64 v44; // rbx
  unsigned __int64 v45; // rax
  __int64 v46; // r8
  unsigned __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // rbx
  unsigned int v50; // eax
  __int64 v51; // rdx
  __int64 v52; // r14
  unsigned int v53; // ebx
  _DWORD *v54; // rcx
  unsigned __int64 v55; // rax
  char v56; // al
  unsigned int v57; // eax
  __int64 v58; // rdx
  char v59; // al
  __int64 v60; // [rsp+10h] [rbp-160h]
  unsigned __int64 v61; // [rsp+10h] [rbp-160h]
  unsigned __int64 v62; // [rsp+10h] [rbp-160h]
  int v63; // [rsp+2Ch] [rbp-144h] BYREF
  __int64 v64[2]; // [rsp+30h] [rbp-140h] BYREF
  _QWORD v65[2]; // [rsp+40h] [rbp-130h] BYREF
  unsigned __int64 *v66; // [rsp+50h] [rbp-120h] BYREF
  __m128i v67; // [rsp+58h] [rbp-118h] BYREF
  __m128i v68; // [rsp+68h] [rbp-108h] BYREF
  __m128i v69; // [rsp+78h] [rbp-F8h]
  const char *v70; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v71; // [rsp+98h] [rbp-D8h]
  __m128i v72; // [rsp+A8h] [rbp-C8h]
  __m128i v73; // [rsp+B8h] [rbp-B8h]
  unsigned __int64 v74; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v75; // [rsp+D8h] [rbp-98h] BYREF
  __m128i v76; // [rsp+E8h] [rbp-88h] BYREF
  __m128i v77; // [rsp+F8h] [rbp-78h] BYREF
  char v78; // [rsp+108h] [rbp-68h] BYREF
  __m128i v79; // [rsp+110h] [rbp-60h] BYREF
  __m128i v80; // [rsp+120h] [rbp-50h] BYREF
  __m128i v81[4]; // [rsp+130h] [rbp-40h] BYREF

  v6 = a4;
  v7 = a2;
  v8 = (__int8 *)a3;
  if ( a4 > 1 && *(_WORD *)a3 == 26990 )
  {
    if ( a4 == 2 || *(_BYTE *)(a3 + 2) != 58 )
    {
      v76.m128i_i8[9] = 1;
      v9 = "ni:<address space>[:<address space>]...";
LABEL_5:
      v76.m128i_i8[8] = 3;
      v74 = (unsigned __int64)v9;
      sub_AE1520(
        (__int64)a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        (__int64 *)v9,
        v75.m128i_i64[0],
        v75.m128i_i32[2],
        v76.m128i_i32[0],
        v76.m128i_i16[4]);
    }
    else
    {
      sub_AE1A40((__int64)&v74, a3 + 3, a4 - 3, 58);
      LOBYTE(v66) = v74;
      v67 = _mm_loadu_si128(&v75);
      v68 = _mm_loadu_si128(&v76);
      v69 = _mm_loadu_si128(&v77);
      if ( (unsigned __int64 *)v77.m128i_i64[0] == &v74 )
      {
        v69.m128i_i64[1] = 1;
        v69.m128i_i64[0] = (__int64)&v66;
      }
      v21 = _mm_loadu_si128(&v79);
      v22 = _mm_loadu_si128(&v80);
      v23 = _mm_loadu_si128(v81);
      LOBYTE(v70) = v78;
      v71 = v21;
      v72 = v22;
      v73 = v23;
      if ( (char *)v81[0].m128i_i64[0] == &v78 )
      {
        v73.m128i_i64[1] = 1;
        v73.m128i_i64[0] = (__int64)&v70;
      }
      v60 = v71.m128i_i64[0];
      if ( v67.m128i_i64[0] == v71.m128i_i64[0] )
      {
LABEL_24:
        *a1 = 1;
      }
      else
      {
        while ( 1 )
        {
          v24 = v67.m128i_i64[0];
          sub_AE1650(v64, v67.m128i_i64[0], v67.m128i_i64[1], &v63);
          v18 = v64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *a1 = v18 | 1;
            return a1;
          }
          v12 = v63;
          if ( !v63 )
            break;
          v13 = *(unsigned int *)(a5 + 8);
          if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            sub_C8D5F0(a5, a5 + 16, v13 + 1, 4);
            v13 = *(unsigned int *)(a5 + 8);
          }
          *(_DWORD *)(*(_QWORD *)a5 + 4 * v13) = v12;
          v14 = v69.m128i_i64[1];
          ++*(_DWORD *)(a5 + 8);
          v15 = sub_C931B0(&v68, v69.m128i_i64[0], v14, 0);
          if ( v15 == -1 )
          {
            v15 = v68.m128i_u64[1];
            v17 = v68.m128i_i64[0];
            v19 = 0;
          }
          else
          {
            v16 = v15 + v14;
            v17 = v68.m128i_i64[0];
            if ( v16 > v68.m128i_i64[1] )
              v16 = v68.m128i_u64[1];
            else
              v18 = v68.m128i_i64[1] - v16;
            v19 = v68.m128i_i64[0] + v16;
            if ( v15 > v68.m128i_i64[1] )
              v15 = v68.m128i_u64[1];
          }
          v67.m128i_i64[0] = v17;
          v67.m128i_i64[1] = v15;
          v68.m128i_i64[0] = v19;
          v68.m128i_i64[1] = v18;
          if ( v17 == v60 )
            goto LABEL_24;
        }
        v50 = sub_C63BB0(v64, v24, v25, v26);
        v52 = v51;
        v53 = v50;
        v64[0] = (__int64)v65;
        sub_AE11D0(v64, "address space 0 cannot be non-integral", (__int64)"");
        sub_C63F00(a1, v64, v53, v52);
        if ( (_QWORD *)v64[0] != v65 )
          j_j___libc_free_0(v64[0], v65[0] + 1LL);
      }
    }
  }
  else
  {
    v10 = *(_BYTE *)a3;
    a2 = *(_BYTE *)a3 & 0xEF;
    if ( (*(_BYTE *)a3 & 0xEF) == 0x66 || v10 == 105 )
    {
      sub_AE2680(a1, v7, (__int64 *)a3, a4);
    }
    else if ( v10 == 97 )
    {
      sub_AE21F0(a1, v7, (const char *)a3, a4);
    }
    else if ( v10 == 112 )
    {
      sub_AE2D40(a1, v7, (const char *)a3, a4);
    }
    else
    {
      if ( a4 )
      {
        v6 = a4 - 1;
        v8 = (__int8 *)(a3 + 1);
      }
      a3 = (unsigned __int8)(v10 - 65);
      switch ( v10 )
      {
        case 'A':
          if ( !v6 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "A<address space>";
            goto LABEL_5;
          }
          v54 = (_DWORD *)(v7 + 4);
          goto LABEL_91;
        case 'E':
        case 'e':
          if ( !v6 )
          {
            *(_BYTE *)v7 = v10 == 69;
            goto LABEL_24;
          }
          v57 = sub_C63BB0(a1, a2, a3, a4);
          v38 = v58;
          v39 = "";
          v74 = (unsigned __int64)&v75.m128i_u64[1];
          v40 = v57;
          v41 = "malformed specification, must be just 'e' or 'E'";
          goto LABEL_67;
        case 'F':
          if ( !v6 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "F<type><abi>";
            goto LABEL_5;
          }
          v10 = *v8;
          if ( *v8 == 105 )
          {
            *(_DWORD *)(v7 + 20) = 0;
          }
          else
          {
            if ( v10 != 110 )
            {
              v27 = "unknown function pointer alignment type '";
LABEL_37:
              v71.m128i_i8[8] = v10;
              v72.m128i_i16[4] = 2051;
              v74 = (unsigned __int64)&v70;
              v75.m128i_i64[1] = (__int64)"'";
              v76.m128i_i16[4] = 770;
              v70 = v27;
              v28 = sub_C63BB0(a1, a2, 770, a4);
              v30 = v29;
              sub_CA0F50(&v66, &v74);
              sub_C63F00(a1, &v66, v28, v30);
              if ( v66 != &v67.m128i_u64[1] )
                j_j___libc_free_0(v66, v67.m128i_i64[1] + 1);
              return a1;
            }
            *(_DWORD *)(v7 + 20) = 1;
          }
          LOBYTE(v70) = 0;
          sub_AE1890(&v74, (__int64)(v8 + 1), v6 - 1, &v70, (__int64)"ABI", 3, 0);
          v55 = v74 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v74 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_92;
          v59 = (char)v70;
          *(_BYTE *)(v7 + 19) = 1;
          *(_BYTE *)(v7 + 18) = v59;
          goto LABEL_24;
        case 'G':
          if ( !v6 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "G<address space>";
            goto LABEL_5;
          }
          v54 = (_DWORD *)(v7 + 12);
          goto LABEL_91;
        case 'P':
          if ( !v6 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "P<address space>";
            goto LABEL_5;
          }
          v54 = (_DWORD *)(v7 + 8);
LABEL_91:
          sub_AE1650(&v74, (__int64)v8, v6, v54);
          v55 = v74 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v74 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_92;
          goto LABEL_24;
        case 'S':
          if ( !v6 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "S<size>";
            goto LABEL_5;
          }
          LOBYTE(v70) = 0;
          sub_AE1890(&v74, (__int64)v8, v6, &v70, (__int64)"stack natural", 13, 0);
          v55 = v74 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v74 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_92:
            *a1 = v55 | 1;
            return a1;
          }
          v56 = (char)v70;
          *(_BYTE *)(v7 + 17) = 1;
          *(_BYTE *)(v7 + 16) = v56;
          goto LABEL_24;
        case 'm':
          if ( !v6 || *v8 != 58 || (v31 = v6 - 1) == 0 )
          {
            v76.m128i_i8[9] = 1;
            v9 = "m:<mangling>";
            goto LABEL_5;
          }
          if ( v31 == 1 )
          {
            switch ( v8[1] )
            {
              case 'a':
                *(_DWORD *)(v7 + 24) = 7;
                goto LABEL_24;
              case 'e':
                *(_DWORD *)(v7 + 24) = 1;
                goto LABEL_24;
              case 'l':
                *(_DWORD *)(v7 + 24) = 5;
                goto LABEL_24;
              case 'm':
                *(_DWORD *)(v7 + 24) = 6;
                goto LABEL_24;
              case 'o':
                *(_DWORD *)(v7 + 24) = 2;
                goto LABEL_24;
              case 'w':
                *(_DWORD *)(v7 + 24) = 3;
                goto LABEL_24;
              case 'x':
                *(_DWORD *)(v7 + 24) = 4;
                goto LABEL_24;
              default:
                break;
            }
          }
          v36 = sub_C63BB0(a1, a2, a3, a4);
          v74 = (unsigned __int64)&v75.m128i_u64[1];
          v38 = v37;
          v39 = "";
          v40 = v36;
          v41 = "unknown mangling mode";
LABEL_67:
          sub_AE11D0((__int64 *)&v74, v41, (__int64)v39);
          sub_C63F00(a1, &v74, v40, v38);
          if ( (unsigned __int64 *)v74 != &v75.m128i_u64[1] )
            j_j___libc_free_0(v74, v75.m128i_i64[1] + 1);
          return a1;
        case 'n':
          sub_AE1A40((__int64)&v74, (__int64)v8, v6, 58);
          v67 = _mm_loadu_si128(&v75);
          LOBYTE(v66) = v74;
          v68 = _mm_loadu_si128(&v76);
          v69 = _mm_loadu_si128(&v77);
          if ( (unsigned __int64 *)v77.m128i_i64[0] == &v74 )
          {
            v69.m128i_i64[1] = 1;
            v69.m128i_i64[0] = (__int64)&v66;
          }
          v32 = _mm_loadu_si128(&v80);
          v71 = _mm_loadu_si128(&v79);
          v33 = _mm_loadu_si128(v81);
          LOBYTE(v70) = v78;
          v72 = v32;
          v73 = v33;
          if ( (char *)v81[0].m128i_i64[0] == &v78 )
          {
            v73.m128i_i64[1] = 1;
            v73.m128i_i64[0] = (__int64)&v70;
          }
          v34 = v71.m128i_i64[0];
          if ( v71.m128i_i64[0] == v67.m128i_i64[0] )
            goto LABEL_24;
          break;
        case 's':
          goto LABEL_24;
        default:
          v27 = "unknown specifier '";
          goto LABEL_37;
      }
      while ( 1 )
      {
        sub_AE1770(v64, v67.m128i_i64[0], v67.m128i_i64[1], &v63, (__int64)"size", 4);
        v35 = v64[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
        v42 = *(_QWORD *)(v7 + 40);
        v43 = v63;
        if ( (unsigned __int64)(v42 + 1) > *(_QWORD *)(v7 + 48) )
        {
          v62 = v64[0] & 0xFFFFFFFFFFFFFFFELL;
          sub_C8D290(v7 + 32, v7 + 56, v42 + 1, 1);
          v42 = *(_QWORD *)(v7 + 40);
          v35 = v62;
        }
        v61 = v35;
        *(_BYTE *)(*(_QWORD *)(v7 + 32) + v42) = v43;
        v44 = v69.m128i_i64[1];
        ++*(_QWORD *)(v7 + 40);
        v45 = sub_C931B0(&v68, v69.m128i_i64[0], v44, 0);
        v46 = v61;
        if ( v45 == -1 )
        {
          v45 = v68.m128i_u64[1];
          v48 = v68.m128i_i64[0];
          v49 = 0;
        }
        else
        {
          v47 = v45 + v44;
          v48 = v68.m128i_i64[0];
          if ( v47 > v68.m128i_i64[1] )
            v47 = v68.m128i_u64[1];
          else
            v46 = v68.m128i_i64[1] - v47;
          v49 = v68.m128i_i64[0] + v47;
          if ( v45 > v68.m128i_i64[1] )
            v45 = v68.m128i_u64[1];
        }
        v67.m128i_i64[0] = v48;
        v67.m128i_i64[1] = v45;
        v68.m128i_i64[0] = v49;
        v68.m128i_i64[1] = v46;
        if ( v34 == v48 )
          goto LABEL_24;
      }
      *a1 = v35 | 1;
    }
  }
  return a1;
}
