// Function: sub_2B50660
// Address: 0x2b50660
//
__int64 __fastcall sub_2B50660(__int64 a1, __int64 a2, unsigned __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // rdx
  int v11; // edx
  unsigned __int8 *v12; // rsi
  int v13; // r11d
  unsigned __int64 v14; // rdi
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  unsigned __int8 *v17; // rcx
  __int64 v18; // rdi
  unsigned int v19; // ebx
  unsigned int v21; // eax
  __int64 v22; // rdx
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned int v25; // edx
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rdx
  __int64 *v31; // rdi
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rcx
  __int64 v35; // rax
  int v36; // edx
  bool v37; // zf
  int v38; // edx
  __int64 v39; // rdx
  int v40; // esi
  bool v41; // cc
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-118h]
  __int64 v50; // [rsp+40h] [rbp-E0h]
  char v51; // [rsp+4Bh] [rbp-D5h]
  int v52; // [rsp+4Ch] [rbp-D4h]
  unsigned __int64 v53; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v55; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v56; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v57; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v58; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v59; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-98h]
  __int64 v61; // [rsp+90h] [rbp-90h] BYREF
  __int64 v62; // [rsp+98h] [rbp-88h]
  __int64 v63; // [rsp+A0h] [rbp-80h]
  unsigned int v64; // [rsp+A8h] [rbp-78h]
  void *s; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v66; // [rsp+B8h] [rbp-68h]
  _DWORD v67[24]; // [rsp+C0h] [rbp-60h] BYREF

  v54 = a3;
  v45 = sub_2B08680(a5, a3);
  if ( v54 > 0x40 )
  {
    sub_C43690((__int64)&v53, 0, 0);
    v56 = a3;
    sub_C43690((__int64)&v55, 0, 0);
    v61 = 0;
    s = v67;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v66 = 0xC00000000LL;
  }
  else
  {
    v53 = 0;
    s = v67;
    v56 = a3;
    v55 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v66 = 0xC00000000LL;
    if ( a3 <= 0xC )
    {
      if ( a3 )
      {
        v21 = 4 * a3;
        if ( 4 * a3 )
        {
          if ( v21 < 8 )
          {
            if ( (v21 & 4) != 0 )
            {
              v67[0] = -1;
              v67[v21 / 4 - 1] = -1;
            }
            else if ( v21 )
            {
              LOBYTE(v67[0]) = -1;
            }
          }
          else
          {
            v22 = v21;
            v23 = v21 - 1;
            *(_QWORD *)((char *)&v67[-2] + v22) = -1;
            if ( v23 >= 8 )
            {
              v24 = v23 & 0xFFFFFFF8;
              v25 = 0;
              do
              {
                v26 = v25;
                v25 += 8;
                *(_QWORD *)((char *)v67 + v26) = -1;
              }
              while ( v25 < v24 );
            }
          }
        }
      }
      LODWORD(v66) = a3;
      if ( (_DWORD)a3 )
      {
LABEL_5:
        v52 = 0;
        v9 = 0;
        v50 = 0;
        v51 = 0;
        while ( 1 )
        {
          v12 = *(unsigned __int8 **)(a2 + 8 * v9);
          v57 = (unsigned __int64)v12;
          if ( a4 && (unsigned __int8)sub_2B0D8B0(v12) || (unsigned int)*v12 - 12 <= 1 )
          {
            v10 = 1LL << v9;
            if ( v54 > 0x40 )
            {
              *(_QWORD *)(v53 + 8LL * ((unsigned int)v9 >> 6)) |= v10;
              v12 = (unsigned __int8 *)v57;
            }
            else
            {
              v53 |= v10;
            }
            v11 = -1;
            if ( *v12 != 13 )
              v11 = v9;
            *((_DWORD *)s + v9) = v11;
            goto LABEL_11;
          }
          if ( !v64 )
            break;
          v13 = 1;
          v14 = 0;
          v15 = (v64 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v16 = v62 + 16LL * v15;
          v17 = *(unsigned __int8 **)v16;
          if ( v12 == *(unsigned __int8 **)v16 )
          {
LABEL_17:
            v18 = 1LL << v9;
            if ( v54 > 0x40 )
              *(_QWORD *)(v53 + 8LL * ((unsigned int)v9 >> 6)) |= v18;
            else
              v53 |= v18;
            v51 = 1;
            *((_DWORD *)s + v9++) = *(_DWORD *)(v16 + 8);
            if ( (unsigned int)a3 == v9 )
              goto LABEL_20;
          }
          else
          {
            while ( v17 != (unsigned __int8 *)-4096LL )
            {
              if ( !v14 && v17 == (unsigned __int8 *)-8192LL )
                v14 = v16;
              v15 = (v64 - 1) & (v13 + v15);
              v16 = v62 + 16LL * v15;
              v17 = *(unsigned __int8 **)v16;
              if ( v12 == *(unsigned __int8 **)v16 )
                goto LABEL_17;
              ++v13;
            }
            if ( !v14 )
              v14 = v16;
            ++v61;
            v33 = v63 + 1;
            v59 = v14;
            if ( 4 * ((int)v63 + 1) >= 3 * v64 )
              goto LABEL_91;
            if ( v64 - HIDWORD(v63) - v33 <= v64 >> 3 )
            {
              v40 = v64;
              goto LABEL_92;
            }
LABEL_76:
            LODWORD(v63) = v33;
            if ( *(_QWORD *)v14 != -4096 )
              --HIDWORD(v63);
            *(_QWORD *)v14 = v12;
            *(_DWORD *)(v14 + 8) = v9;
            v34 = *(_QWORD *)(v57 + 8);
            if ( v34 != a5 )
            {
              v35 = sub_DFD060(*(__int64 **)(a1 + 3296), 38, a5, v34);
              v37 = v36 == 1;
              v38 = 1;
              if ( !v37 )
                v38 = v52;
              v52 = v38;
              if ( __OFADD__(v35, v50) )
              {
                v41 = v35 <= 0;
                v42 = 0x8000000000000000LL;
                if ( !v41 )
                  v42 = 0x7FFFFFFFFFFFFFFFLL;
                v50 = v42;
              }
              else
              {
                v50 += v35;
              }
            }
            if ( !a4 )
            {
              v39 = 1LL << v9;
              if ( v56 > 0x40 )
                *(_QWORD *)(v55 + 8LL * ((unsigned int)v9 >> 6)) |= v39;
              else
                v55 |= v39;
            }
            *((_DWORD *)s + v9) = v9;
LABEL_11:
            if ( (unsigned int)a3 == ++v9 )
              goto LABEL_20;
          }
        }
        ++v61;
        v59 = 0;
LABEL_91:
        v40 = 2 * v64;
LABEL_92:
        sub_D39D40((__int64)&v61, v40);
        sub_22B1A50((__int64)&v61, (__int64 *)&v57, &v59);
        v12 = (unsigned __int8 *)v57;
        v14 = v59;
        v33 = v63 + 1;
        goto LABEL_76;
      }
      v51 = 0;
      v50 = 0;
      goto LABEL_42;
    }
  }
  sub_C8D5F0((__int64)&s, v67, a3, 4u, v7, v8);
  if ( 4 * a3 )
    memset(s, 255, 4 * a3);
  LODWORD(v66) = a3;
  if ( (_DWORD)a3 )
    goto LABEL_5;
  v50 = 0;
  v51 = 0;
LABEL_20:
  v19 = v56;
  if ( v56 > 0x40 )
  {
    if ( v19 == (unsigned int)sub_C444A0((__int64)&v55) )
      goto LABEL_22;
    goto LABEL_43;
  }
LABEL_42:
  if ( !v55 )
    goto LABEL_22;
LABEL_43:
  v27 = sub_2B0F120(*(__int64 **)(a1 + 3296), a5, v45, (__int64)&v55, 0);
  if ( __OFADD__(v27, v50) )
  {
    v41 = v27 <= 0;
    v43 = 0x8000000000000000LL;
    if ( !v41 )
      v43 = 0x7FFFFFFFFFFFFFFFLL;
    v50 = v43;
LABEL_22:
    if ( !a4 )
      goto LABEL_23;
    goto LABEL_45;
  }
  v50 += v27;
  if ( !a4 )
    goto LABEL_23;
LABEL_45:
  v28 = v54;
  v58 = v54;
  if ( v54 <= 0x40 )
  {
    v29 = v53;
LABEL_47:
    v30 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v28) & ~v29;
    if ( !v28 )
      v30 = 0;
    v57 = v30;
    goto LABEL_50;
  }
  sub_C43780((__int64)&v57, (const void **)&v53);
  v28 = v58;
  if ( v58 <= 0x40 )
  {
    v29 = v57;
    goto LABEL_47;
  }
  sub_C43D10((__int64)&v57);
  v28 = v58;
  v30 = v57;
LABEL_50:
  v60 = v28;
  v59 = v30;
  v31 = *(__int64 **)(a1 + 3296);
  v58 = 0;
  v50 = sub_2B0F120(v31, a5, v45, (__int64)&v59, 0);
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
LABEL_23:
  if ( v51 )
  {
    v32 = sub_DFBC30(*(__int64 **)(a1 + 3296), 7, v45, (__int64)s, (unsigned int)v66, 0, 0, 0, 0, 0, 0);
    if ( __OFADD__(v32, v50) )
    {
      v41 = v32 <= 0;
      v44 = 0x8000000000000000LL;
      if ( !v41 )
        v44 = 0x7FFFFFFFFFFFFFFFLL;
      v50 = v44;
    }
    else
    {
      v50 += v32;
    }
  }
  if ( s != v67 )
    _libc_free((unsigned __int64)s);
  sub_C7D6A0(v62, 16LL * v64, 8);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  return v50;
}
