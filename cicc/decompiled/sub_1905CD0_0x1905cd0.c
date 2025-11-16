// Function: sub_1905CD0
// Address: 0x1905cd0
//
__int64 __fastcall sub_1905CD0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __m128i *v4; // rsi
  bool v5; // bl
  _QWORD *v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // ecx
  __m128i *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r13
  unsigned __int32 v15; // eax
  __int64 v16; // r12
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // r13
  __int64 v20; // r14
  void *v21; // rax
  void *v22; // rdx
  void *v23; // r12
  __int64 v24; // rsi
  unsigned __int8 v25; // al
  __int64 v26; // rbx
  const void **v27; // r12
  const void *v28; // rdi
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  char v31; // al
  unsigned __int32 v32; // edx
  __int64 v33; // rax
  const void **v34; // rdx
  void *v35; // rax
  __int64 v36; // rsi
  unsigned __int8 v37; // al
  int v38; // edx
  __int64 v39; // r13
  __int64 v40; // rsi
  __int64 v41; // r12
  int v42; // r9d
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // [rsp+8h] [rbp-188h]
  __int64 v47; // [rsp+20h] [rbp-170h]
  _BYTE *v48; // [rsp+28h] [rbp-168h]
  void *v49; // [rsp+30h] [rbp-160h]
  __int64 v50; // [rsp+38h] [rbp-158h]
  char v51; // [rsp+4Fh] [rbp-141h] BYREF
  __int64 v52; // [rsp+50h] [rbp-140h] BYREF
  unsigned int v53; // [rsp+58h] [rbp-138h]
  char v54; // [rsp+5Ch] [rbp-134h]
  __int64 v55; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v56; // [rsp+68h] [rbp-128h]
  __m128i v57; // [rsp+70h] [rbp-120h] BYREF
  void (__fastcall *v58)(__m128i *, __m128i *, __int64); // [rsp+80h] [rbp-110h]
  __int64 (__fastcall *v59)(__int64, __int64, __int64 *); // [rsp+88h] [rbp-108h]
  const void **v60; // [rsp+90h] [rbp-100h] BYREF
  void *v61; // [rsp+98h] [rbp-F8h] BYREF
  __int64 v62; // [rsp+A0h] [rbp-F0h]
  __int64 v63; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int v64; // [rsp+B8h] [rbp-D8h]
  __int64 v65; // [rsp+C0h] [rbp-D0h]
  unsigned int v66; // [rsp+C8h] [rbp-C8h]
  __m128i v67; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v69; // [rsp+E8h] [rbp-A8h]

  result = *(_QWORD *)(a1 + 40);
  v46 = *(_QWORD *)(a1 + 32);
  v47 = result - 16;
  if ( result != v46 )
  {
    while ( 1 )
    {
      v3 = (__int64)&v67;
      v4 = (__m128i *)a1;
      sub_1904880((__int64)&v67);
      if ( *(_DWORD *)(v47 - 8) <= 0x40u )
        break;
      v4 = &v67;
      v3 = v47 - 16;
      v5 = sub_16A5220(v47 - 16, (const void **)&v67);
      if ( v5 )
      {
        v6 = (_QWORD *)v47;
        if ( *(_DWORD *)(v47 + 8) <= 0x40u )
          goto LABEL_125;
LABEL_8:
        v3 = v47;
        v4 = (__m128i *)&v68;
        v5 = sub_16A5220(v47, (const void **)&v68);
      }
LABEL_9:
      if ( v69 > 0x40 )
      {
        v3 = v68;
        if ( v68 )
          j_j___libc_free_0_0(v68);
      }
      if ( v67.m128i_i32[2] > 0x40u )
      {
        v3 = v67.m128i_i64[0];
        if ( v67.m128i_i64[0] )
          j_j___libc_free_0_0(v67.m128i_i64[0]);
      }
      if ( v5 )
      {
        v7 = *(_QWORD *)(v47 - 24);
        v58 = 0;
        v48 = (_BYTE *)v7;
        switch ( *(_BYTE *)(v7 + 16) )
        {
          case '$':
          case '&':
          case '(':
            v67.m128i_i64[0] = v7;
            v8 = _mm_loadu_si128(&v67);
            v58 = (void (__fastcall *)(__m128i *, __m128i *, __int64))sub_1903E70;
            v59 = sub_1903F10;
            v57 = v8;
            break;
          case '%':
          case '\'':
          case ')':
          case '*':
          case '+':
          case ',':
          case '-':
          case '.':
          case '/':
          case '0':
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
          case ':':
          case ';':
          case '<':
          case '=':
          case '>':
          case 'A':
          case 'B':
          case 'C':
          case 'D':
          case 'E':
          case 'F':
          case 'G':
          case 'H':
          case 'I':
          case 'J':
          case 'K':
          case 'L':
            v30 = _mm_loadu_si128(&v67);
            v58 = (void (__fastcall *)(__m128i *, __m128i *, __int64))sub_1903DD0;
            v59 = sub_1903E40;
            v57 = v30;
            break;
          case '?':
          case '@':
            v67.m128i_i64[0] = v7;
            v29 = _mm_loadu_si128(&v67);
            v58 = (void (__fastcall *)(__m128i *, __m128i *, __int64))sub_1903EA0;
            v59 = sub_1903ED0;
            v57 = v29;
            break;
        }
        v67.m128i_i64[0] = (__int64)&v68;
        v67.m128i_i64[1] = 0x400000000LL;
        if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
        {
          v9 = *(_QWORD *)(v7 - 8);
          v50 = v9 + 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
        }
        else
        {
          v50 = v7;
          v9 = v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
        }
        if ( v50 != v9 )
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)v9;
            if ( *(_BYTE *)(*(_QWORD *)v9 + 16LL) > 0x17u )
            {
              v10 = *(unsigned int *)(a1 + 24);
              if ( (_DWORD)v10 )
              {
                v3 = (unsigned int)(v10 - 1);
                v4 = *(__m128i **)(a1 + 8);
                v11 = v3 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
                v12 = &v4[v11];
                v13 = v12->m128i_i64[0];
                if ( v19 == v12->m128i_i64[0] )
                {
LABEL_24:
                  if ( v12 != &v4[v10] )
                  {
                    v14 = *(_QWORD *)(a1 + 32) + 40LL * v12->m128i_u32[2];
                    v15 = v67.m128i_u32[2];
                    if ( v67.m128i_i32[2] < (unsigned __int32)v67.m128i_i32[3] )
                    {
LABEL_26:
                      v16 = v67.m128i_i64[0] + 32LL * v15;
                      if ( v16 )
                      {
                        v17 = *(_DWORD *)(v14 + 16);
                        *(_DWORD *)(v16 + 8) = v17;
                        if ( v17 > 0x40 )
                        {
                          v4 = (__m128i *)(v14 + 8);
                          v3 = v16;
                          sub_16A4FD0(v16, (const void **)(v14 + 8));
                        }
                        else
                        {
                          *(_QWORD *)v16 = *(_QWORD *)(v14 + 8);
                        }
                        v18 = *(_DWORD *)(v14 + 32);
                        *(_DWORD *)(v16 + 24) = v18;
                        if ( v18 > 0x40 )
                        {
                          v4 = (__m128i *)(v14 + 24);
                          v3 = v16 + 16;
                          sub_16A4FD0(v16 + 16, (const void **)(v14 + 24));
                        }
                        else
                        {
                          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v14 + 24);
                        }
                        v15 = v67.m128i_u32[2];
                      }
                      v67.m128i_i32[2] = v15 + 1;
                      goto LABEL_33;
                    }
LABEL_111:
                    v3 = (__int64)&v67;
                    v4 = 0;
                    sub_1905050((__int64)&v67, 0);
                    v15 = v67.m128i_u32[2];
                    goto LABEL_26;
                  }
                }
                else
                {
                  v38 = 1;
                  while ( v13 != -8 )
                  {
                    v42 = v38 + 1;
                    v11 = v3 & (v38 + v11);
                    v12 = &v4[v11];
                    v13 = v12->m128i_i64[0];
                    if ( v19 == v12->m128i_i64[0] )
                      goto LABEL_24;
                    v38 = v42;
                  }
                }
              }
              v14 = *(_QWORD *)(a1 + 40);
              v15 = v67.m128i_u32[2];
              if ( v67.m128i_i32[2] < (unsigned __int32)v67.m128i_i32[3] )
                goto LABEL_26;
              goto LABEL_111;
            }
            v20 = v19 + 24;
            v49 = *(void **)(v19 + 32);
            v21 = sub_16982C0();
            v22 = v49;
            v23 = v21;
            if ( v49 == v21 )
            {
              v36 = *(_QWORD *)(v19 + 40);
              v37 = *(_BYTE *)(v36 + 26) & 7;
              if ( v37 <= 1u )
              {
LABEL_37:
                sub_1904850((__int64)&v63);
                goto LABEL_38;
              }
              v24 = v36 + 8;
              if ( v37 != 3 )
              {
                v24 = v19 + 32;
                goto LABEL_96;
              }
            }
            else
            {
              v24 = v19 + 32;
              v25 = *(_BYTE *)(v19 + 50) & 7;
              if ( v25 <= 1u )
                goto LABEL_37;
              if ( v25 != 3 )
                goto LABEL_67;
            }
            if ( (*(_BYTE *)(v24 + 18) & 8) == 0 )
              goto LABEL_107;
            v31 = *(_BYTE *)(*(_QWORD *)v48 + 8LL);
            if ( v31 == 16 )
              v31 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v48 + 16LL) + 8LL);
            if ( (unsigned __int8)(v31 - 1) <= 5u || v48[16] == 76 )
            {
              if ( !sub_15F24C0((__int64)v48) )
                goto LABEL_37;
              v22 = *(void **)(v19 + 32);
              v24 = v19 + 32;
            }
            else
            {
LABEL_107:
              v24 = v19 + 32;
            }
            if ( v23 != v22 )
            {
LABEL_67:
              sub_16986C0(&v61, (__int64 *)v24);
              if ( v23 == v61 )
                goto LABEL_97;
              goto LABEL_68;
            }
LABEL_96:
            sub_169C6E0(&v61, v24);
            if ( v23 == v61 )
            {
LABEL_97:
              if ( (unsigned int)sub_169EBA0(&v61, 0) )
              {
LABEL_98:
                sub_1904850((__int64)&v63);
                sub_19058A0(a1, (__int64)v48, &v63);
                if ( v66 > 0x40 && v65 )
                  j_j___libc_free_0_0(v65);
                if ( v64 > 0x40 && v63 )
                  j_j___libc_free_0_0(v63);
                if ( v23 == v61 )
                {
                  v43 = v62;
                  if ( v62 )
                  {
                    v44 = 32LL * *(_QWORD *)(v62 - 8);
                    v45 = v62 + v44;
                    if ( v62 != v62 + v44 )
                    {
                      do
                      {
                        v45 -= 32;
                        sub_127D120((_QWORD *)(v45 + 8));
                      }
                      while ( v43 != v45 );
                    }
                    j_j_j___libc_free_0_0(v43 - 8);
                  }
                }
                else
                {
                  sub_1698460((__int64)&v61);
                }
                goto LABEL_44;
              }
              goto LABEL_69;
            }
LABEL_68:
            if ( (unsigned int)sub_169D440((__int64)&v61, 0) )
              goto LABEL_98;
LABEL_69:
            if ( (unsigned int)sub_14A9E40((__int64)&v60, v20) != 1 )
              goto LABEL_98;
            v53 = dword_4FAE5E0 + 1;
            if ( (unsigned int)(dword_4FAE5E0 + 1) > 0x40 )
              sub_16A4EF0((__int64)&v52, 0, 0);
            else
              v52 = 0;
            v54 = 0;
            sub_169E1A0(v20, (__int64)&v52, 0, &v51);
            v56 = v53;
            if ( v53 > 0x40 )
              sub_16A4FD0((__int64)&v55, (const void **)&v52);
            else
              v55 = v52;
            v3 = (__int64)&v63;
            v4 = (__m128i *)&v55;
            sub_1589870((__int64)&v63, &v55);
            v32 = v67.m128i_u32[2];
            if ( v67.m128i_i32[2] >= (unsigned __int32)v67.m128i_i32[3] )
            {
              v3 = (__int64)&v67;
              v4 = 0;
              sub_1905050((__int64)&v67, 0);
              v32 = v67.m128i_u32[2];
            }
            v33 = v67.m128i_i64[0] + 32LL * v32;
            if ( v33 )
            {
              *(_DWORD *)(v33 + 8) = v64;
              *(_QWORD *)v33 = v63;
              v64 = 0;
              *(_DWORD *)(v33 + 24) = v66;
              *(_QWORD *)(v33 + 16) = v65;
              ++v67.m128i_i32[2];
            }
            else
            {
              v67.m128i_i32[2] = v32 + 1;
              if ( v66 > 0x40 )
              {
                v3 = v65;
                if ( v65 )
                  j_j___libc_free_0_0(v65);
              }
            }
            if ( v64 > 0x40 )
            {
              v3 = v63;
              if ( v63 )
                j_j___libc_free_0_0(v63);
            }
            if ( v56 > 0x40 )
            {
              v3 = v55;
              if ( v55 )
                j_j___libc_free_0_0(v55);
            }
            if ( v53 > 0x40 )
            {
              v3 = v52;
              if ( v52 )
                j_j___libc_free_0_0(v52);
            }
            if ( v23 == v61 )
            {
              v39 = v62;
              if ( v62 )
              {
                v40 = 32LL * *(_QWORD *)(v62 - 8);
                v41 = v62 + v40;
                if ( v62 != v62 + v40 )
                {
                  do
                  {
                    v41 -= 32;
                    sub_127D120((_QWORD *)(v41 + 8));
                  }
                  while ( v39 != v41 );
                  v40 = 32LL * *(_QWORD *)(v39 - 8);
                }
                v4 = (__m128i *)(v40 + 8);
                v3 = v39 - 8;
                j_j_j___libc_free_0_0(v39 - 8);
              }
LABEL_33:
              v9 += 24;
              if ( v50 == v9 )
                goto LABEL_89;
            }
            else
            {
              v3 = (__int64)&v61;
              v9 += 24;
              sub_1698460((__int64)&v61);
              if ( v50 == v9 )
              {
LABEL_89:
                v34 = (const void **)v67.m128i_i64[0];
                v35 = (void *)v67.m128i_u32[2];
                goto LABEL_90;
              }
            }
          }
        }
        v34 = (const void **)&v68;
        v35 = 0;
LABEL_90:
        v60 = v34;
        v61 = v35;
        if ( !v58 )
          sub_4263D6(v3, v4, v34);
        v59((__int64)&v63, (__int64)&v57, (__int64 *)&v60);
LABEL_38:
        sub_19058A0(a1, (__int64)v48, &v63);
        if ( v66 > 0x40 && v65 )
          j_j___libc_free_0_0(v65);
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
LABEL_44:
        v26 = v67.m128i_i64[0];
        v27 = (const void **)(v67.m128i_i64[0] + 32LL * v67.m128i_u32[2]);
        if ( (const void **)v67.m128i_i64[0] != v27 )
        {
          do
          {
            v27 -= 4;
            if ( *((_DWORD *)v27 + 6) > 0x40u )
            {
              v28 = v27[2];
              if ( v28 )
                j_j___libc_free_0_0(v28);
            }
            if ( *((_DWORD *)v27 + 2) > 0x40u && *v27 )
              j_j___libc_free_0_0(*v27);
          }
          while ( (const void **)v26 != v27 );
          v27 = (const void **)v67.m128i_i64[0];
        }
        if ( v27 != (const void **)&v68 )
          _libc_free((unsigned __int64)v27);
        if ( v58 )
          v58(&v57, &v57, 3);
      }
      result = v47 - 24;
      if ( v46 == v47 - 24 )
        return result;
      v47 -= 40;
    }
    v5 = 0;
    if ( *(_QWORD *)(v47 - 16) != v67.m128i_i64[0] )
      goto LABEL_9;
    v6 = (_QWORD *)v47;
    if ( *(_DWORD *)(v47 + 8) <= 0x40u )
    {
LABEL_125:
      v5 = *v6 == v68;
      goto LABEL_9;
    }
    goto LABEL_8;
  }
  return result;
}
