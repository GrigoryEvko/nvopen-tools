// Function: sub_25146E0
// Address: 0x25146e0
//
void __fastcall sub_25146E0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  size_t v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdi
  _QWORD *v16; // r15
  _QWORD *v17; // r12
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  unsigned __int64 v20; // r14
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r15
  unsigned __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // [rsp+28h] [rbp-D8h]
  _QWORD *v36; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v37; // [rsp+30h] [rbp-D0h] BYREF
  size_t v38; // [rsp+38h] [rbp-C8h]
  __int64 v39; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v40; // [rsp+50h] [rbp-B0h] BYREF
  size_t v41; // [rsp+58h] [rbp-A8h]
  _BYTE v42[16]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int8 *v43; // [rsp+70h] [rbp-90h] BYREF
  size_t v44; // [rsp+78h] [rbp-88h]
  _BYTE v45[16]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int8 *v46; // [rsp+90h] [rbp-70h] BYREF
  size_t v47; // [rsp+98h] [rbp-68h]
  __int64 v48; // [rsp+A0h] [rbp-60h]
  __int64 v49; // [rsp+A8h] [rbp-58h]
  __int64 v50; // [rsp+B0h] [rbp-50h]
  __int64 v51; // [rsp+B8h] [rbp-48h]
  unsigned __int8 **v52; // [rsp+C0h] [rbp-40h]

  sub_25072F0((__int64 *)&v37, byte_3F871B3);
  v3 = sub_904010(*(_QWORD *)a1, "\tNode");
  v4 = sub_CB5A80(v3, a2);
  sub_904010(v4, " [shape=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_904010(*(_QWORD *)a1, "none,");
    v5 = v38;
    if ( !v38 )
      goto LABEL_3;
  }
  else
  {
    sub_904010(*(_QWORD *)a1, "record,");
    v5 = v38;
    if ( !v38 )
      goto LABEL_3;
  }
  v25 = sub_CB6200(*(_QWORD *)a1, v37, v5);
  sub_904010(v25, ",");
LABEL_3:
  sub_904010(*(_QWORD *)a1, "label=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    v6 = *(_QWORD *)(a2 + 40);
    v7 = 1;
    v8 = v6 + 8LL * *(unsigned int *)(a2 + 48);
    v9 = 0;
    if ( v6 != v8 )
    {
      do
      {
        v6 += 8;
        ++v9;
        if ( v6 == v8 )
        {
          v7 = v9;
          goto LABEL_9;
        }
      }
      while ( v9 != 64 );
      v7 = 65;
    }
LABEL_9:
    v10 = sub_904010(*(_QWORD *)a1, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"");
    v11 = sub_904010(v10, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"");
    v12 = sub_CB59D0(v11, v7);
    sub_904010(v12, "\">");
  }
  else
  {
    sub_904010(*(_QWORD *)a1, "\"{");
  }
  v13 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v52 = &v43;
    v51 = 0x100000000LL;
    v45[0] = 0;
    v43 = v45;
    v44 = 0;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    v46 = (unsigned __int8 *)&unk_49DD210;
    sub_CB5980((__int64)&v46, 0, 0, 0);
    (*(void (__fastcall **)(unsigned __int64, _QWORD, unsigned __int8 **))(*(_QWORD *)a2 + 16LL))(a2, 0, &v46);
    v46 = (unsigned __int8 *)&unk_49DD210;
    sub_CB5840((__int64)&v46);
    v14 = sub_CB6200(v13, v43, v44);
    sub_904010(v14, "</td>");
    if ( v43 != v45 )
      j_j___libc_free_0((unsigned __int64)v43);
  }
  else
  {
    v52 = &v43;
    v51 = 0x100000000LL;
    v45[0] = 0;
    v46 = (unsigned __int8 *)&unk_49DD210;
    v43 = v45;
    v44 = 0;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    sub_CB5980((__int64)&v46, 0, 0, 0);
    (*(void (__fastcall **)(unsigned __int64, _QWORD, unsigned __int8 **))(*(_QWORD *)a2 + 16LL))(a2, 0, &v46);
    v46 = (unsigned __int8 *)&unk_49DD210;
    sub_CB5840((__int64)&v46);
    sub_C67200((__int64 *)&v46, (__int64)&v43);
    sub_CB6200(v13, v46, v47);
    sub_2240A30((unsigned __int64 *)&v46);
    sub_2240A30((unsigned __int64 *)&v43);
  }
  sub_25072F0((__int64 *)&v40, byte_3F871B3);
  if ( v41 )
  {
    v29 = sub_904010(*(_QWORD *)a1, "|");
    sub_C67200((__int64 *)&v46, (__int64)&v40);
    sub_CB6200(v29, v46, v47);
    sub_2240A30((unsigned __int64 *)&v46);
  }
  sub_25072F0((__int64 *)&v43, byte_3F871B3);
  if ( v44 )
  {
    v28 = sub_904010(*(_QWORD *)a1, "|");
    sub_C67200((__int64 *)&v46, (__int64)&v43);
    sub_CB6200(v28, v46, v47);
    sub_2240A30((unsigned __int64 *)&v46);
  }
  if ( v43 != v45 )
    j_j___libc_free_0((unsigned __int64)v43);
  if ( v40 != v42 )
    j_j___libc_free_0((unsigned __int64)v40);
  v42[0] = 0;
  v40 = v42;
  v51 = 0x100000000LL;
  v41 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v46 = (unsigned __int8 *)&unk_49DD210;
  v52 = &v40;
  sub_CB5980((__int64)&v46, 0, 0, 0);
  if ( (unsigned __int8)sub_25146B0(a1, (__int64)&v46) )
  {
    if ( *(_BYTE *)(a1 + 16) || (sub_904010(*(_QWORD *)a1, "|"), *(_BYTE *)(a1 + 16)) )
    {
      sub_CB6200(*(_QWORD *)a1, v40, v41);
    }
    else
    {
      v26 = sub_904010(*(_QWORD *)a1, "{");
      v27 = sub_CB6200(v26, v40, v41);
      sub_904010(v27, "}");
    }
  }
  v15 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) )
    sub_904010(v15, "</tr></table>>");
  else
    sub_904010(v15, "}\"");
  sub_904010(*(_QWORD *)a1, "];\n");
  v16 = *(_QWORD **)(a2 + 40);
  v17 = &v16[*(unsigned int *)(a2 + 48)];
  if ( v17 != v16 )
  {
    v35 = v16 + 64;
    while ( 1 )
    {
      v20 = *v16 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v20 )
      {
        v21 = *(_QWORD *)a1;
        v43 = v45;
        v44 = 0;
        v45[0] = 0;
        v22 = *(_QWORD *)(v21 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v22) > 4 )
        {
          *(_DWORD *)v22 = 1685016073;
          *(_BYTE *)(v22 + 4) = 101;
          *(_QWORD *)(v21 + 32) += 5LL;
        }
        else
        {
          v21 = sub_CB6200(v21, "\tNode", 5u);
        }
        sub_CB5A80(v21, a2);
        v18 = *(_QWORD *)a1;
        v19 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v19 <= 7u )
        {
          v18 = sub_CB6200(v18, " -> Node", 8u);
        }
        else
        {
          *v19 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v18 + 32) += 8LL;
        }
        sub_CB5A80(v18, v20);
        if ( v44 )
        {
          v23 = sub_904010(*(_QWORD *)a1, "[");
          v24 = sub_CB6200(v23, v43, v44);
          sub_904010(v24, "]");
        }
        sub_904010(*(_QWORD *)a1, ";\n");
        if ( v43 != v45 )
          j_j___libc_free_0((unsigned __int64)v43);
      }
      if ( v17 == ++v16 )
        break;
      if ( v16 == v35 )
      {
        v36 = v17;
        do
        {
          v30 = *v16 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v30 )
          {
            sub_25072F0((__int64 *)&v43, byte_3F871B3);
            v31 = sub_904010(*(_QWORD *)a1, "\tNode");
            sub_CB5A80(v31, a2);
            v32 = sub_904010(*(_QWORD *)a1, " -> Node");
            sub_CB5A80(v32, v30);
            if ( v44 )
            {
              v33 = sub_904010(*(_QWORD *)a1, "[");
              v34 = sub_CB6200(v33, v43, v44);
              sub_904010(v34, "]");
            }
            sub_904010(*(_QWORD *)a1, ";\n");
            if ( v43 != v45 )
              j_j___libc_free_0((unsigned __int64)v43);
          }
          ++v16;
        }
        while ( v36 != v16 );
        break;
      }
    }
  }
  v46 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v46);
  if ( v40 != v42 )
    j_j___libc_free_0((unsigned __int64)v40);
  if ( v37 != (unsigned __int8 *)&v39 )
    j_j___libc_free_0((unsigned __int64)v37);
}
