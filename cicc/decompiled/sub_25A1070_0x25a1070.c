// Function: sub_25A1070
// Address: 0x25a1070
//
void __fastcall sub_25A1070(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  size_t v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // edx
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  char *v14; // rsi
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int8 *v17; // rax
  char *v18; // rax
  __int64 v19; // rdx
  size_t v20; // rdx
  unsigned __int8 *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  signed __int64 v24; // r12
  __m128i v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  bool v30; // cl
  bool v31; // zf
  __int64 v32; // r13
  unsigned __int64 v33; // r13
  size_t v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r12
  unsigned __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  char *v48; // rsi
  __int64 v49; // r12
  __int64 v50; // r12
  __int64 v51; // rax
  __int64 v52; // rax
  size_t v53; // [rsp+10h] [rbp-120h]
  size_t v55; // [rsp+30h] [rbp-100h]
  __int64 v56; // [rsp+38h] [rbp-F8h]
  __m128i v57; // [rsp+40h] [rbp-F0h] BYREF
  __m128i v58; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int8 *v59; // [rsp+60h] [rbp-D0h] BYREF
  size_t v60; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v61; // [rsp+80h] [rbp-B0h] BYREF
  size_t v62; // [rsp+88h] [rbp-A8h]
  char v63; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int8 *v64; // [rsp+A0h] [rbp-90h] BYREF
  size_t v65; // [rsp+A8h] [rbp-88h]
  _BYTE v66[16]; // [rsp+B0h] [rbp-80h] BYREF
  unsigned __int8 *v67; // [rsp+C0h] [rbp-70h] BYREF
  size_t v68; // [rsp+C8h] [rbp-68h]
  _QWORD v69[12]; // [rsp+D0h] [rbp-60h] BYREF

  sub_253C590((__int64 *)&v59, byte_3F871B3);
  v3 = sub_904010(*(_QWORD *)a1, "\tNode");
  v4 = sub_CB5A80(v3, a2);
  sub_904010(v4, " [shape=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_904010(*(_QWORD *)a1, "none,");
    v5 = v60;
    if ( !v60 )
      goto LABEL_3;
  }
  else
  {
    sub_904010(*(_QWORD *)a1, "record,");
    v5 = v60;
    if ( !v60 )
      goto LABEL_3;
  }
  v36 = sub_CB6200(*(_QWORD *)a1, v59, v5);
  sub_904010(v36, ",");
LABEL_3:
  sub_904010(*(_QWORD *)a1, "label=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    v6 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)a2 + 16LL))(a2);
    v7 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)a2 + 24LL))(a2);
    if ( v6 == v7 )
    {
      v10 = 1;
    }
    else
    {
      v8 = v6 + 8;
      v9 = 0;
      do
      {
        ++v9;
        if ( v7 == v8 )
        {
          v10 = v9;
          goto LABEL_9;
        }
        v8 += 8;
      }
      while ( v9 != 64 );
      v10 = 65;
    }
LABEL_9:
    v11 = sub_904010(*(_QWORD *)a1, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"");
    v12 = sub_904010(v11, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"");
    v13 = sub_CB59D0(v12, v10);
    v14 = "\">";
    sub_904010(v13, "\">");
    if ( *(_BYTE *)(a1 + 16) )
    {
      v15 = *(_QWORD *)a1;
LABEL_11:
      v16 = a2 - 104;
LABEL_12:
      v17 = sub_250CBE0((__int64 *)(v16 + 72), (__int64)v14);
      v18 = (char *)sub_BD5D20((__int64)v17);
      if ( v18 )
      {
        v67 = (unsigned __int8 *)v69;
        sub_2539970((__int64 *)&v67, v18, (__int64)&v18[v19]);
        v20 = v68;
        v21 = v67;
      }
      else
      {
        v21 = (unsigned __int8 *)v69;
        LOBYTE(v69[0]) = 0;
        v20 = 0;
        v67 = (unsigned __int8 *)v69;
        v68 = 0;
      }
      v22 = sub_CB6200(v15, v21, v20);
      sub_904010(v22, "</td>");
      sub_2240A30((unsigned __int64 *)&v67);
      goto LABEL_15;
    }
    v15 = *(_QWORD *)a1;
LABEL_54:
    v16 = a2 - 104;
    goto LABEL_55;
  }
  v14 = "\"{";
  sub_904010(*(_QWORD *)a1, "\"{");
  v15 = *(_QWORD *)a1;
  v16 = 0;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( !a2 )
      goto LABEL_12;
    goto LABEL_11;
  }
  if ( a2 )
    goto LABEL_54;
LABEL_55:
  v46 = sub_250CBE0((__int64 *)(v16 + 72), (__int64)v14);
  v48 = (char *)sub_BD5D20((__int64)v46);
  if ( v48 )
  {
    v64 = v66;
    sub_2539970((__int64 *)&v64, v48, (__int64)&v48[v47]);
  }
  else
  {
    v66[0] = 0;
    v64 = v66;
    v65 = 0;
  }
  sub_C67200((__int64 *)&v67, (__int64)&v64);
  sub_CB6200(v15, v67, v68);
  sub_2240A30((unsigned __int64 *)&v67);
  sub_2240A30((unsigned __int64 *)&v64);
LABEL_15:
  sub_253C590((__int64 *)&v61, byte_3F871B3);
  if ( v62 )
  {
    v50 = sub_904010(*(_QWORD *)a1, "|");
    sub_C67200((__int64 *)&v67, (__int64)&v61);
    sub_CB6200(v50, v67, v68);
    sub_2240A30((unsigned __int64 *)&v67);
  }
  sub_253C590((__int64 *)&v64, byte_3F871B3);
  if ( v65 )
  {
    v49 = sub_904010(*(_QWORD *)a1, "|");
    sub_C67200((__int64 *)&v67, (__int64)&v64);
    sub_CB6200(v49, v67, v68);
    sub_2240A30((unsigned __int64 *)&v67);
  }
  sub_2240A30((unsigned __int64 *)&v64);
  sub_2240A30((unsigned __int64 *)&v61);
  v61 = (unsigned __int8 *)&v63;
  v69[3] = 0x100000000LL;
  v69[4] = &v61;
  v63 = 0;
  v68 = 0;
  v62 = 0;
  memset(v69, 0, 24);
  v67 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v67, 0, 0, 0);
  if ( (unsigned __int8)sub_256A650(a1, (__int64)&v67, a2) )
  {
    if ( *(_BYTE *)(a1 + 16) || (sub_904010(*(_QWORD *)a1, "|"), *(_BYTE *)(a1 + 16)) )
    {
      sub_CB6200(*(_QWORD *)a1, v61, v62);
    }
    else
    {
      v51 = sub_904010(*(_QWORD *)a1, "{");
      v52 = sub_CB6200(v51, v61, v62);
      sub_904010(v52, "}");
    }
  }
  v23 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) )
    sub_904010(v23, "</tr></table>>");
  else
    sub_904010(v23, "}\"");
  v24 = 0;
  sub_904010(*(_QWORD *)a1, "];\n");
  v25.m128i_i64[0] = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)a2 + 16LL))(a2);
  v57 = v25;
  v56 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)a2 + 24LL))(a2);
  if ( v57.m128i_i64[0] != v56 )
  {
    do
    {
      v32 = **(_QWORD **)(a1 + 8);
      if ( v32 != sub_25A1010((__int64)&v57) )
      {
        v58 = _mm_loadu_si128(&v57);
        v33 = sub_25A1010((__int64)&v58);
        if ( v33 )
        {
          sub_253C590((__int64 *)&v64, byte_3F871B3);
          v34 = v65;
          if ( v64 != v66 )
          {
            v53 = v65;
            j_j___libc_free_0((unsigned __int64)v64);
            v34 = v53;
          }
          if ( v34 )
          {
            sub_253C590((__int64 *)&v64, byte_3F871B3);
            v26 = sub_904010(*(_QWORD *)a1, "\tNode");
            sub_CB5A80(v26, a2);
            v27 = sub_904010(*(_QWORD *)a1, ":s");
            sub_CB59F0(v27, v24);
          }
          else
          {
            sub_253C590((__int64 *)&v64, byte_3F871B3);
            v35 = sub_904010(*(_QWORD *)a1, "\tNode");
            sub_CB5A80(v35, a2);
          }
          v28 = sub_904010(*(_QWORD *)a1, " -> Node");
          sub_CB5A80(v28, v33);
          if ( v65 )
          {
            v42 = sub_904010(*(_QWORD *)a1, "[");
            v43 = sub_CB6200(v42, v64, v65);
            sub_904010(v43, "]");
          }
          sub_904010(*(_QWORD *)a1, ";\n");
          if ( v64 != v66 )
            j_j___libc_free_0((unsigned __int64)v64);
        }
      }
      v29 = v57.m128i_i64[0] + 8;
      v30 = (_DWORD)v24 != 63;
      v31 = v56 == v57.m128i_i64[0] + 8;
      v57.m128i_i64[0] += 8;
      ++v24;
    }
    while ( !v31 && v30 );
    if ( v56 != v29 )
    {
      do
      {
        v39 = **(_QWORD **)(a1 + 8);
        if ( v39 != sub_25A1010((__int64)&v57) )
        {
          v58 = _mm_loadu_si128(&v57);
          v40 = sub_25A1010((__int64)&v58);
          if ( v40 )
          {
            sub_253C590((__int64 *)&v64, byte_3F871B3);
            v55 = v65;
            sub_2240A30((unsigned __int64 *)&v64);
            sub_253C590((__int64 *)&v64, byte_3F871B3);
            v41 = sub_904010(*(_QWORD *)a1, "\tNode");
            sub_CB5A80(v41, a2);
            if ( v55 )
            {
              v37 = sub_904010(*(_QWORD *)a1, ":s");
              sub_CB59F0(v37, 64);
            }
            v38 = sub_904010(*(_QWORD *)a1, " -> Node");
            sub_CB5A80(v38, v40);
            if ( v65 )
            {
              v44 = sub_904010(*(_QWORD *)a1, "[");
              v45 = sub_CB6200(v44, v64, v65);
              sub_904010(v45, "]");
            }
            sub_904010(*(_QWORD *)a1, ";\n");
            sub_2240A30((unsigned __int64 *)&v64);
          }
        }
        v57.m128i_i64[0] += 8;
      }
      while ( v56 != v57.m128i_i64[0] );
    }
  }
  v67 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v67);
  sub_2240A30((unsigned __int64 *)&v61);
  sub_2240A30((unsigned __int64 *)&v59);
}
