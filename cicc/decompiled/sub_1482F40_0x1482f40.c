// Function: sub_1482F40
// Address: 0x1482f40
//
__int64 __fastcall sub_1482F40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        _QWORD *a9)
{
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // r15
  int v25; // r11d
  unsigned int v26; // eax
  __int64 **v27; // r13
  __int64 *v28; // rcx
  bool v30; // zf
  char v31; // al
  int v32; // eax
  int v33; // ecx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 *v40; // r8
  int v41; // r10d
  unsigned int v42; // eax
  _QWORD *v43; // rcx
  __int64 v44; // rdi
  int v45; // eax
  char v46; // al
  int v47; // ecx
  int v48; // eax
  __int64 v49; // rax
  int v50; // eax
  __int64 v52; // [rsp+10h] [rbp-D0h]
  __int64 v53; // [rsp+10h] [rbp-D0h]
  unsigned int v54; // [rsp+18h] [rbp-C8h]
  unsigned int v55; // [rsp+18h] [rbp-C8h]
  int v56; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v60; // [rsp+38h] [rbp-A8h]
  _QWORD *v61; // [rsp+38h] [rbp-A8h]
  __int64 *v62; // [rsp+48h] [rbp-98h] BYREF
  __int64 v63; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v64; // [rsp+58h] [rbp-88h]
  __int64 v65; // [rsp+60h] [rbp-80h] BYREF
  __int64 v66; // [rsp+68h] [rbp-78h] BYREF
  unsigned int v67; // [rsp+70h] [rbp-70h]
  __int64 *v68; // [rsp+80h] [rbp-60h] BYREF
  __int64 v69; // [rsp+88h] [rbp-58h] BYREF
  _DWORD v70[20]; // [rsp+90h] [rbp-50h] BYREF

  v10 = a1;
  v11 = *a4;
  if ( !*(_WORD *)(*a4 + 24) )
  {
    v60 = 0;
    v12 = 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 32);
      ++v12;
      v54 = *(_DWORD *)(a6 + 8);
      v14 = v13 + 24;
      if ( v54 > 0x40 )
      {
        v52 = v13;
        v32 = sub_16A57B0(a6);
        v13 = v52;
        if ( v54 - v32 > 0x40 )
          goto LABEL_6;
        v15 = **(_QWORD ***)a6;
      }
      else
      {
        v15 = *(_QWORD **)a6;
      }
      if ( v15 == (_QWORD *)1 )
      {
        v55 = *(_DWORD *)(a3 + 8);
        if ( v55 > 0x40 )
        {
          v53 = v13;
          v45 = sub_16A57B0(a3);
          v13 = v53;
          if ( v55 - v45 <= 0x40 && !**(_QWORD **)a3 )
          {
LABEL_26:
            if ( *(_DWORD *)(v13 + 32) <= 0x40u )
            {
              v46 = v60;
              if ( !*(_QWORD *)(v13 + 24) )
                v46 = 1;
              v60 = v46;
            }
            else
            {
              v56 = *(_DWORD *)(v13 + 32);
              v30 = v56 == (unsigned int)sub_16A57B0(v14);
              v31 = v60;
              if ( v30 )
                v31 = 1;
              v60 = v31;
            }
            goto LABEL_7;
          }
        }
        else if ( !*(_QWORD *)a3 )
        {
          goto LABEL_26;
        }
      }
LABEL_6:
      v60 = 1;
LABEL_7:
      sub_16A7B50(&v68, a6, v14);
      sub_16A7200(a3, &v68);
      if ( (unsigned int)v69 > 0x40 && v68 )
        j_j___libc_free_0_0(v68);
      v16 = v12;
      v11 = a4[v12];
      if ( *(_WORD *)(v11 + 24) )
      {
        v10 = a1;
        goto LABEL_12;
      }
    }
  }
  v60 = 0;
  v16 = 0;
  v12 = 0;
LABEL_12:
  v17 = a6;
  if ( a5 != v16 )
  {
    while ( 1 )
    {
      v18 = a4[v16];
      if ( *(_WORD *)(v18 + 24) == 5 )
      {
        v19 = **(_QWORD **)(v18 + 32);
        if ( !*(_WORD *)(v19 + 24) )
        {
          sub_16A7B50(&v63, v17, *(_QWORD *)(v19 + 32) + 24LL);
          v34 = *(_QWORD *)(v18 + 40);
          v35 = *(_QWORD *)(v18 + 32);
          if ( v34 == 2 && (v36 = *(_QWORD *)(v35 + 8), *(_WORD *)(v36 + 24) == 4) )
          {
            v60 |= sub_1482F40(
                     v10,
                     a2,
                     a3,
                     *(_QWORD *)(v36 + 32),
                     *(_QWORD *)(v36 + 40),
                     (unsigned int)&v63,
                     (__int64)a9);
          }
          else
          {
            v69 = 0x400000000LL;
            v68 = (__int64 *)v70;
            sub_145C5B0((__int64)&v68, (_BYTE *)(v35 + 8), (_BYTE *)(v35 + 8 * v34));
            v65 = sub_147EE30(a9, &v68, 0, 0, a7, a8);
            v67 = v64;
            if ( v64 <= 0x40 )
            {
              v37 = *(_DWORD *)(v10 + 24);
              v66 = v63;
              if ( v37 )
                goto LABEL_40;
LABEL_58:
              ++*(_QWORD *)v10;
LABEL_59:
              v37 *= 2;
              goto LABEL_60;
            }
            sub_16A4FD0(&v66, &v63);
            v37 = *(_DWORD *)(v10 + 24);
            if ( !v37 )
              goto LABEL_58;
LABEL_40:
            v38 = v65;
            v39 = *(_QWORD *)(v10 + 8);
            v40 = 0;
            v41 = 1;
            v42 = (v37 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
            v43 = (_QWORD *)(v39 + 24LL * v42);
            v44 = *v43;
            if ( *v43 == v65 )
            {
LABEL_41:
              if ( v67 > 0x40 && v66 )
              {
                v61 = v43;
                j_j___libc_free_0_0(v66);
                v43 = v61;
              }
              sub_16A7200(v43 + 1, &v63);
              v60 = 1;
            }
            else
            {
              while ( v44 != -8 )
              {
                if ( !v40 && v44 == -16 )
                  v40 = v43;
                v42 = (v37 - 1) & (v41 + v42);
                v43 = (_QWORD *)(v39 + 24LL * v42);
                v44 = *v43;
                if ( v65 == *v43 )
                  goto LABEL_41;
                ++v41;
              }
              v50 = *(_DWORD *)(v10 + 16);
              if ( !v40 )
                v40 = v43;
              ++*(_QWORD *)v10;
              v47 = v50 + 1;
              if ( 4 * (v50 + 1) >= 3 * v37 )
                goto LABEL_59;
              if ( v37 - *(_DWORD *)(v10 + 20) - v47 <= v37 >> 3 )
              {
LABEL_60:
                sub_1466B60(v10, v37);
                sub_145F4D0(v10, &v65, &v62);
                v40 = v62;
                v38 = v65;
                v47 = *(_DWORD *)(v10 + 16) + 1;
              }
              *(_DWORD *)(v10 + 16) = v47;
              if ( *v40 != -8 )
                --*(_DWORD *)(v10 + 20);
              *v40 = v38;
              *((_DWORD *)v40 + 4) = v67;
              v40[1] = v66;
              sub_1458920(a2, v40);
            }
            if ( v68 != (__int64 *)v70 )
              _libc_free((unsigned __int64)v68);
          }
          if ( v64 > 0x40 && v63 )
            j_j___libc_free_0_0(v63);
          goto LABEL_22;
        }
      }
      v20 = *(_DWORD *)(v17 + 8);
      v68 = (__int64 *)v18;
      v70[0] = v20;
      if ( v20 > 0x40 )
      {
        sub_16A4FD0(&v69, v17);
        v21 = *(_DWORD *)(v10 + 24);
        if ( !v21 )
        {
LABEL_33:
          ++*(_QWORD *)v10;
          goto LABEL_34;
        }
      }
      else
      {
        v21 = *(_DWORD *)(v10 + 24);
        v69 = *(_QWORD *)v17;
        if ( !v21 )
          goto LABEL_33;
      }
      v22 = v68;
      v23 = *(_QWORD *)(v10 + 8);
      v24 = 0;
      v25 = 1;
      v26 = (v21 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
      v27 = (__int64 **)(v23 + 24LL * v26);
      v28 = *v27;
      if ( *v27 != v68 )
        break;
LABEL_18:
      if ( v70[0] > 0x40u && v69 )
        j_j___libc_free_0_0(v69);
      sub_16A7200(v27 + 1, v17);
      v60 = 1;
LABEL_22:
      v16 = ++v12;
      if ( v12 == a5 )
        return v60;
    }
    while ( v28 != (__int64 *)-8LL )
    {
      if ( !v24 && v28 == (__int64 *)-16LL )
        v24 = (__int64)v27;
      v26 = (v21 - 1) & (v25 + v26);
      v27 = (__int64 **)(v23 + 24LL * v26);
      v28 = *v27;
      if ( v68 == *v27 )
        goto LABEL_18;
      ++v25;
    }
    v48 = *(_DWORD *)(v10 + 16);
    if ( !v24 )
      v24 = (__int64)v27;
    ++*(_QWORD *)v10;
    v33 = v48 + 1;
    if ( 4 * (v48 + 1) >= 3 * v21 )
    {
LABEL_34:
      v21 *= 2;
    }
    else if ( v21 - *(_DWORD *)(v10 + 20) - v33 > v21 >> 3 )
    {
      goto LABEL_72;
    }
    sub_1466B60(v10, v21);
    sub_145F4D0(v10, (__int64 *)&v68, &v65);
    v24 = v65;
    v22 = v68;
    v33 = *(_DWORD *)(v10 + 16) + 1;
LABEL_72:
    *(_DWORD *)(v10 + 16) = v33;
    if ( *(_QWORD *)v24 != -8 )
      --*(_DWORD *)(v10 + 20);
    *(_QWORD *)v24 = v22;
    *(_DWORD *)(v24 + 16) = v70[0];
    *(_QWORD *)(v24 + 8) = v69;
    v49 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v49 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, 0, 8);
      v49 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v49) = *(_QWORD *)v24;
    ++*(_DWORD *)(a2 + 8);
    goto LABEL_22;
  }
  return v60;
}
