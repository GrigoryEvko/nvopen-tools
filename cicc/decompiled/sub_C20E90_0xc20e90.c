// Function: sub_C20E90
// Address: 0xc20e90
//
__int64 __fastcall sub_C20E90(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rax
  void *v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // eax
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  void *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rdi
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __m128i *v26; // rdx
  __m128i si128; // xmm0
  __int64 v28; // rdi
  __int64 v29; // rdi
  _BYTE *v30; // rax
  void *v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  _BYTE *v34; // rax
  __int64 v36; // rax
  __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v40[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v41; // [rsp+50h] [rbp-50h] BYREF
  __int64 v42; // [rsp+58h] [rbp-48h]
  _QWORD v43[8]; // [rsp+60h] [rbp-40h] BYREF

  v2 = 0;
  v3 = a2;
  v4 = *(_QWORD *)(a1 + 400);
  v38 = *(_QWORD *)(a1 + 408);
  if ( v38 == v4 )
    goto LABEL_28;
  do
  {
    switch ( *(_DWORD *)v4 )
    {
      case 0:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "InvalidSection", (__int64)"");
        break;
      case 1:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "ProfileSummarySection", (__int64)"");
        break;
      case 2:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "NameTableSection", (__int64)"");
        break;
      case 3:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "ProfileSymbolListSection", (__int64)"");
        break;
      case 4:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "FuncOffsetTableSection", (__int64)"");
        break;
      case 5:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "FunctionMetadata", (__int64)"");
        break;
      case 6:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "CSNameTableSection", (__int64)"");
        break;
      case 0x20:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "LBRProfileSection", (__int64)"");
        break;
      default:
        v39[0] = (__int64)v40;
        sub_C1EB20(v39, "UnknownSection", (__int64)"");
        break;
    }
    v5 = sub_CB6200(a2, v39[0], v39[1]);
    v6 = *(void **)(v5 + 32);
    v7 = v5;
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xAu )
    {
      v7 = sub_CB6200(v5, " - Offset: ", 11);
    }
    else
    {
      qmemcpy(v6, " - Offset: ", 11);
      *(_QWORD *)(v5 + 32) += 11LL;
    }
    v8 = sub_CB59D0(v7, *(_QWORD *)(v4 + 16));
    v9 = *(_QWORD **)(v8 + 32);
    v10 = v8;
    if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 7u )
    {
      v10 = sub_CB6200(v8, ", Size: ", 8);
    }
    else
    {
      *v9 = 0x203A657A6953202CLL;
      *(_QWORD *)(v8 + 32) += 8LL;
    }
    v11 = sub_CB59D0(v10, *(_QWORD *)(v4 + 24));
    v13 = *(_QWORD *)(v11 + 32);
    v14 = v11;
    if ( (unsigned __int64)(*(_QWORD *)(v11 + 24) - v13) <= 8 )
    {
      v14 = sub_CB6200(v11, ", Flags: ", 9);
    }
    else
    {
      *(_BYTE *)(v13 + 8) = 32;
      *(_QWORD *)v13 = 0x3A7367616C46202CLL;
      *(_QWORD *)(v11 + 32) += 9LL;
    }
    v42 = 0;
    LOBYTE(v43[0]) = 0;
    v41 = v43;
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      sub_2241490(&v41, "{compressed,", 12, v12);
      v15 = *(_QWORD *)(v4 + 8);
      v16 = v15;
      if ( (v15 & 2) == 0 )
        goto LABEL_12;
    }
    else
    {
      sub_2241490(&v41, "{", 1, v12);
      v15 = *(_QWORD *)(v4 + 8);
      v16 = v15;
      if ( (v15 & 2) == 0 )
      {
LABEL_12:
        v17 = *(_DWORD *)v4;
        if ( *(_DWORD *)v4 == 2 )
          goto LABEL_72;
        if ( v17 <= 2 )
          goto LABEL_49;
        if ( v17 == 4 )
          goto LABEL_78;
        if ( v17 != 5 )
          goto LABEL_18;
        if ( (v15 & 0x100000000LL) == 0 )
          goto LABEL_17;
        goto LABEL_62;
      }
    }
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v42) <= 4 )
      goto LABEL_93;
    sub_2241490(&v41, "flat,", 5, v16);
    v17 = *(_DWORD *)v4;
    if ( *(_DWORD *)v4 == 2 )
    {
      v16 = *(_QWORD *)(v4 + 8);
LABEL_72:
      if ( (v16 & 0x200000000LL) != 0 )
      {
        sub_C1EBD0((__int64)&v41, "fixlenmd5,");
      }
      else if ( (v16 & 0x100000000LL) != 0 )
      {
        sub_C1EBD0((__int64)&v41, "md5,");
      }
      if ( *(_DWORD *)v4 != 2 )
LABEL_94:
        BUG();
      if ( (*(_BYTE *)(v4 + 12) & 4) != 0 )
        sub_C1EBD0((__int64)&v41, "uniq,");
      goto LABEL_18;
    }
    if ( v17 <= 2 )
    {
LABEL_49:
      if ( v17 == 1 )
      {
        if ( (*(_BYTE *)(v4 + 12) & 1) != 0 )
        {
          sub_C1EBD0((__int64)&v41, "partial,");
          if ( *(_DWORD *)v4 != 1 )
            goto LABEL_94;
        }
        v36 = *(_QWORD *)(v4 + 8);
        if ( (v36 & 0x200000000LL) != 0 )
        {
          sub_C1EBD0((__int64)&v41, "context,");
          if ( *(_DWORD *)v4 != 1 )
            goto LABEL_94;
          v36 = *(_QWORD *)(v4 + 8);
        }
        if ( (v36 & 0x1000000000LL) != 0 )
        {
          sub_C1EBD0((__int64)&v41, "preInlined,");
          if ( *(_DWORD *)v4 != 1 )
            goto LABEL_94;
          v36 = *(_QWORD *)(v4 + 8);
        }
        if ( (v36 & 0x400000000LL) != 0 )
          sub_C1EBD0((__int64)&v41, "fs-discriminator,");
      }
      goto LABEL_18;
    }
    if ( v17 == 4 )
    {
LABEL_78:
      if ( (*(_BYTE *)(v4 + 12) & 1) != 0 )
        sub_C1EBD0((__int64)&v41, "ordered,");
      goto LABEL_18;
    }
    if ( v17 != 5 )
      goto LABEL_18;
    if ( (*(_QWORD *)(v4 + 8) & 0x100000000LL) == 0 )
    {
LABEL_17:
      if ( (*(_BYTE *)(v4 + 12) & 2) == 0 )
        goto LABEL_18;
LABEL_64:
      sub_C1EBD0((__int64)&v41, "attr,");
      goto LABEL_18;
    }
LABEL_62:
    sub_C1EBD0((__int64)&v41, "probe,");
    if ( *(_DWORD *)v4 != 5 )
      goto LABEL_94;
    if ( (*(_BYTE *)(v4 + 12) & 2) != 0 )
      goto LABEL_64;
LABEL_18:
    v18 = (char *)v41 + v42 - 1;
    if ( *v18 == 44 )
    {
      *v18 = 125;
    }
    else
    {
      if ( v42 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_93:
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v41, "}", 1, v16);
    }
    v19 = sub_CB6200(v14, v41, v42);
    v20 = *(_BYTE **)(v19 + 32);
    if ( *(_BYTE **)(v19 + 24) == v20 )
    {
      sub_CB6200(v19, "\n", 1);
    }
    else
    {
      *v20 = 10;
      ++*(_QWORD *)(v19 + 32);
    }
    if ( v41 != v43 )
      j_j___libc_free_0(v41, v43[0] + 1LL);
    if ( (_QWORD *)v39[0] != v40 )
      j_j___libc_free_0(v39[0], v40[0] + 1LL);
    v2 += *(_QWORD *)(v4 + 24);
    v4 += 40;
  }
  while ( v38 != v4 );
  v4 = *(_QWORD *)(a1 + 400);
LABEL_28:
  v21 = *(void **)(a2 + 32);
  v22 = *(_QWORD *)(v4 + 16);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v21 <= 0xCu )
  {
    v23 = sub_CB6200(a2, "Header Size: ", 13);
  }
  else
  {
    v23 = a2;
    qmemcpy(v21, "Header Size: ", 13);
    *(_QWORD *)(a2 + 32) += 13LL;
  }
  v24 = sub_CB59D0(v23, v22);
  v25 = *(_BYTE **)(v24 + 32);
  if ( *(_BYTE **)(v24 + 24) == v25 )
  {
    sub_CB6200(v24, "\n", 1);
  }
  else
  {
    *v25 = 10;
    ++*(_QWORD *)(v24 + 32);
  }
  v26 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 <= 0x14u )
  {
    v28 = sub_CB6200(a2, "Total Sections Size: ", 21);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F64F10);
    v26[1].m128i_i32[0] = 979729001;
    v28 = a2;
    v26[1].m128i_i8[4] = 32;
    *v26 = si128;
    *(_QWORD *)(a2 + 32) += 21LL;
  }
  v29 = sub_CB59D0(v28, v2);
  v30 = *(_BYTE **)(v29 + 32);
  if ( *(_BYTE **)(v29 + 24) == v30 )
  {
    sub_CB6200(v29, "\n", 1);
  }
  else
  {
    *v30 = 10;
    ++*(_QWORD *)(v29 + 32);
  }
  v31 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v31 <= 0xAu )
  {
    v3 = sub_CB6200(a2, "File Size: ", 11);
  }
  else
  {
    qmemcpy(v31, "File Size: ", 11);
    *(_QWORD *)(a2 + 32) += 11LL;
  }
  v32 = sub_C20E50(a1);
  v33 = sub_CB59D0(v3, v32);
  v34 = *(_BYTE **)(v33 + 32);
  if ( *(_BYTE **)(v33 + 24) == v34 )
  {
    sub_CB6200(v33, "\n", 1);
  }
  else
  {
    *v34 = 10;
    ++*(_QWORD *)(v33 + 32);
  }
  return 1;
}
