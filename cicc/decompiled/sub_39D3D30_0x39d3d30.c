// Function: sub_39D3D30
// Address: 0x39d3d30
//
void __fastcall sub_39D3D30(__int64 *a1, __int64 a2)
{
  _DWORD *v2; // r13
  __int64 v5; // rax
  __int64 v6; // r8
  _BYTE *v7; // r9
  __int64 *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 (*v11)(); // rdx
  __int64 (*v12)(); // rax
  __int64 v13; // r15
  unsigned int v14; // r14d
  _BYTE *v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  char v18; // al
  __int64 v19; // rdi
  _WORD *v20; // rdx
  __int16 v21; // ax
  char *v22; // rsi
  size_t v23; // rdx
  void *v24; // rdi
  __int64 v25; // rdi
  __m128i *v26; // rdx
  unsigned __int8 *v27; // rax
  unsigned __int64 v28; // r12
  __int64 v29; // rdi
  _BYTE *v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  _WORD *v34; // rdx
  __int64 v35; // rdi
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rdi
  _DWORD *v42; // rdx
  __int64 v43; // rdi
  _BYTE *v44; // rax
  __int64 v45; // rax
  __int64 *v46; // r10
  __int64 v47; // r14
  _QWORD *v48; // r13
  __int64 v49; // r11
  __int64 *v50; // rbx
  __int64 v51; // rdi
  _WORD *v52; // rdx
  __int64 v53; // rdi
  void *v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rdx
  size_t v57; // [rsp+8h] [rbp-78h]
  __int64 *v58; // [rsp+10h] [rbp-70h]
  __int64 v59; // [rsp+18h] [rbp-68h]
  __int64 v60; // [rsp+20h] [rbp-60h]
  char v61; // [rsp+28h] [rbp-58h]
  __int64 v62; // [rsp+30h] [rbp-50h]
  __int64 v63; // [rsp+30h] [rbp-50h]
  unsigned int v64; // [rsp+38h] [rbp-48h]
  __int64 *v65; // [rsp+38h] [rbp-48h]
  unsigned __int64 v66[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = 0;
  v5 = sub_1E15F70(a2);
  v8 = *(__int64 **)(v5 + 16);
  v9 = *(_QWORD *)(v5 + 40);
  v58 = (__int64 *)v5;
  v10 = *v8;
  v62 = v9;
  v11 = *(__int64 (**)())(*v8 + 112);
  if ( v11 != sub_1D00B10 )
  {
    v2 = (_DWORD *)((__int64 (__fastcall *)(__int64 *))v11)(v8);
    v10 = *v8;
  }
  v12 = *(__int64 (**)())(v10 + 40);
  v59 = 0;
  if ( v12 != sub_1D00B00 )
    v59 = ((__int64 (__fastcall *)(__int64 *))v12)(v8);
  v13 = 0;
  v14 = 0;
  v66[0] = 0x2000000000000001LL;
  v61 = sub_1E17EA0(a2, a2, (__int64)v11, v9, v6, v7);
  v64 = *(_DWORD *)(a2 + 40);
  if ( v64 )
  {
    while ( 1 )
    {
      v17 = (_BYTE *)(v13 + *(_QWORD *)(a2 + 32));
      if ( *v17 )
        break;
      v18 = v17[3];
      if ( (v18 & 0x10) == 0 || (v18 & 0x20) != 0 )
        break;
      if ( v14 )
      {
        v19 = *a1;
        v20 = *(_WORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v20 <= 1u )
        {
          sub_16E7EE0(v19, ", ", 2u);
        }
        else
        {
          *v20 = 8236;
          *(_QWORD *)(v19 + 24) += 2LL;
        }
      }
      v13 += 40;
      v15 = (_BYTE *)sub_1E17F60(a2, v14, v66, v62);
      v16 = v14++;
      sub_39D38F0(a1, a2, v16, v2, v61, v15, 0);
      if ( v64 == v14 )
        goto LABEL_73;
    }
    if ( !v14 )
      goto LABEL_16;
LABEL_73:
    v55 = *a1;
    v56 = *(_QWORD *)(*a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v56) <= 2 )
    {
      sub_16E7EE0(v55, " = ", 3u);
    }
    else
    {
      *(_BYTE *)(v56 + 2) = 32;
      *(_WORD *)v56 = 15648;
      *(_QWORD *)(v55 + 24) += 3LL;
    }
  }
  else
  {
LABEL_16:
    v14 = 0;
  }
  v21 = *(_WORD *)(a2 + 46);
  if ( (v21 & 1) != 0 )
  {
    v53 = *a1;
    v54 = *(void **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v54 <= 0xBu )
    {
      sub_16E7EE0(v53, "frame-setup ", 0xCu);
    }
    else
    {
      qmemcpy(v54, "frame-setup ", 12);
      *(_QWORD *)(v53 + 24) += 12LL;
    }
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 2) != 0 )
  {
    sub_1263B40(*a1, "frame-destroy ");
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x10) != 0 )
  {
    sub_1263B40(*a1, "nnan ");
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x20) != 0 )
  {
    sub_1263B40(*a1, "ninf ");
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x40) != 0 )
  {
    v41 = *a1;
    v42 = *(_DWORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v42 <= 3u )
    {
      sub_16E7EE0(v41, "nsz ", 4u);
    }
    else
    {
      *v42 = 544895854;
      *(_QWORD *)(v41 + 24) += 4LL;
    }
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x80u) != 0 )
  {
    v39 = *a1;
    v40 = *(_QWORD *)(*a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v40) <= 4 )
    {
      sub_16E7EE0(v39, "arcp ", 5u);
    }
    else
    {
      *(_DWORD *)v40 = 1885565537;
      *(_BYTE *)(v40 + 4) = 32;
      *(_QWORD *)(v39 + 24) += 5LL;
    }
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x100) != 0 )
  {
    v37 = *a1;
    v38 = *(_QWORD *)(*a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v38) <= 8 )
    {
      sub_16E7EE0(v37, "contract ", 9u);
    }
    else
    {
      *(_BYTE *)(v38 + 8) = 32;
      *(_QWORD *)v38 = 0x74636172746E6F63LL;
      *(_QWORD *)(v37 + 24) += 9LL;
    }
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x200) != 0 )
  {
    sub_1263B40(*a1, "afn ");
    v21 = *(_WORD *)(a2 + 46);
  }
  if ( (v21 & 0x400) != 0 )
  {
    v35 = *a1;
    v36 = *(_QWORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v36 <= 7u )
    {
      sub_16E7EE0(v35, "reassoc ", 8u);
    }
    else
    {
      *v36 = 0x20636F7373616572LL;
      *(_QWORD *)(v35 + 24) += 8LL;
    }
  }
  v22 = (char *)(*(_QWORD *)(v59 + 24)
               + *(unsigned int *)(*(_QWORD *)(v59 + 16) + 4LL * **(unsigned __int16 **)(a2 + 16)));
  if ( !v22 )
  {
LABEL_34:
    if ( v64 <= v14 )
      goto LABEL_35;
    goto LABEL_45;
  }
  v60 = *a1;
  v23 = strlen(v22);
  v24 = *(void **)(v60 + 24);
  if ( *(_QWORD *)(v60 + 16) - (_QWORD)v24 >= v23 )
  {
    if ( v23 )
    {
      v57 = v23;
      memcpy(v24, v22, v23);
      *(_QWORD *)(v60 + 24) += v57;
    }
    goto LABEL_34;
  }
  sub_16E7EE0(v60, v22, v23);
  if ( v64 <= v14 )
  {
LABEL_35:
    if ( !*(_QWORD *)(a2 + 64) )
      goto LABEL_40;
    goto LABEL_36;
  }
LABEL_45:
  v29 = *a1;
  v30 = *(_BYTE **)(*a1 + 24);
  if ( (unsigned __int64)v30 >= *(_QWORD *)(*a1 + 16) )
  {
    sub_16E7DE0(v29, 32);
  }
  else
  {
    *(_QWORD *)(v29 + 24) = v30 + 1;
    *v30 = 32;
  }
  while ( 1 )
  {
    v31 = (_BYTE *)sub_1E17F60(a2, v14, v66, v62);
    v32 = v14++;
    sub_39D38F0(a1, a2, v32, v2, v61, v31, 1u);
    if ( v14 == v64 )
      break;
    v33 = *a1;
    v34 = *(_WORD **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v34 <= 1u )
    {
      sub_16E7EE0(v33, ", ", 2u);
    }
    else
    {
      *v34 = 8236;
      *(_QWORD *)(v33 + 24) += 2LL;
    }
  }
  if ( *(_QWORD *)(a2 + 64) )
  {
    v43 = *a1;
    v44 = *(_BYTE **)(*a1 + 24);
    if ( (unsigned __int64)v44 >= *(_QWORD *)(*a1 + 16) )
    {
      sub_16E7DE0(v43, 44);
    }
    else
    {
      *(_QWORD *)(v43 + 24) = v44 + 1;
      *v44 = 44;
    }
LABEL_36:
    v25 = *a1;
    v26 = *(__m128i **)(*a1 + 24);
    if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v26 <= 0xFu )
    {
      sub_16E7EE0(v25, " debug-location ", 0x10u);
    }
    else
    {
      *v26 = _mm_load_si128((const __m128i *)&xmmword_42EB2D0);
      *(_QWORD *)(v25 + 24) += 16LL;
    }
    v27 = (unsigned __int8 *)sub_15C70A0(a2 + 64);
    sub_1556260(v27, *a1, a1[1], 0);
  }
LABEL_40:
  if ( *(_BYTE *)(a2 + 49) )
  {
    sub_1263B40(*a1, " :: ");
    v45 = sub_15E0530(*v58);
    v46 = *(__int64 **)(a2 + 56);
    v47 = v58[7];
    v48 = (_QWORD *)v45;
    v65 = &v46[*(unsigned __int8 *)(a2 + 49)];
    if ( v46 != v65 )
    {
      v49 = *v46;
      v50 = v46 + 1;
      while ( 1 )
      {
        sub_1E343B0(v49, *a1, a1[1], (__int64)(a1 + 4), v48, v47, v59);
        if ( v65 == v50 )
          break;
        v51 = *a1;
        v49 = *v50;
        v52 = *(_WORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v52 <= 1u )
        {
          v63 = *v50++;
          sub_16E7EE0(v51, ", ", 2u);
          v49 = v63;
        }
        else
        {
          ++v50;
          *v52 = 8236;
          *(_QWORD *)(v51 + 24) += 2LL;
        }
      }
    }
  }
  v28 = v66[0];
  if ( (v66[0] & 1) == 0 )
  {
    if ( v66[0] )
    {
      _libc_free(*(_QWORD *)v66[0]);
      j_j___libc_free_0(v28);
    }
  }
}
