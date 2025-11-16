// Function: sub_23B61B0
// Address: 0x23b61b0
//
void __fastcall sub_23B61B0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // r15
  unsigned __int64 v22; // r13
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  __int64 v25; // r15
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // r12

  *a1 = &unk_4A160B8;
  v2 = a1[5];
  if ( v2 )
  {
    v3 = sub_904010(v2, "<script>var coll = document.getElementsByClassName(\"collapsible\");");
    v4 = sub_904010(v3, "var i;");
    v5 = sub_904010(v4, "for (i = 0; i < coll.length; i++) {");
    v6 = sub_904010(v5, "coll[i].addEventListener(\"click\", function() {");
    v7 = sub_904010(v6, " this.classList.toggle(\"active\");");
    v8 = sub_904010(v7, " var content = this.nextElementSibling;");
    v9 = sub_904010(v8, " if (content.style.display === \"block\"){");
    v10 = sub_904010(v9, " content.style.display = \"none\";");
    v11 = sub_904010(v10, " }");
    v12 = sub_904010(v11, " else {");
    v13 = sub_904010(v12, " content.style.display= \"block\";");
    v14 = sub_904010(v13, " }");
    v15 = sub_904010(v14, " });");
    v16 = sub_904010(v15, " }");
    v17 = sub_904010(v16, "</script>");
    v18 = sub_904010(v17, "</body>");
    sub_904010(v18, "</html>\n");
    v19 = (__int64 *)a1[5];
    if ( v19[4] != v19[2] )
    {
      sub_CB5AE0(v19);
      v19 = (__int64 *)a1[5];
    }
    sub_CB7080((__int64)v19, (__int64)"</html>\n");
    v20 = a1[5];
    if ( v20 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
    v21 = a1[2];
    v22 = a1[1];
    *a1 = &unk_4A15E98;
    if ( v21 != v22 )
    {
      do
      {
        sub_23B5F50(v22 + 24);
        v23 = *(unsigned __int64 **)(v22 + 8);
        v24 = *(unsigned __int64 **)v22;
        if ( v23 != *(unsigned __int64 **)v22 )
        {
          do
          {
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              j_j___libc_free_0(*v24);
            v24 += 4;
          }
          while ( v23 != v24 );
          v24 = *(unsigned __int64 **)v22;
        }
        if ( v24 )
          j_j___libc_free_0((unsigned __int64)v24);
        v22 += 48LL;
      }
      while ( v21 != v22 );
LABEL_15:
      v22 = a1[1];
    }
  }
  else
  {
    v25 = a1[2];
    v22 = a1[1];
    *a1 = &unk_4A15E98;
    if ( v25 != v22 )
    {
      do
      {
        sub_23B5F50(v22 + 24);
        v26 = *(unsigned __int64 **)(v22 + 8);
        v27 = *(unsigned __int64 **)v22;
        if ( v26 != *(unsigned __int64 **)v22 )
        {
          do
          {
            if ( (unsigned __int64 *)*v27 != v27 + 2 )
              j_j___libc_free_0(*v27);
            v27 += 4;
          }
          while ( v26 != v27 );
          v27 = *(unsigned __int64 **)v22;
        }
        if ( v27 )
          j_j___libc_free_0((unsigned __int64)v27);
        v22 += 48LL;
      }
      while ( v25 != v22 );
      goto LABEL_15;
    }
  }
  if ( v22 )
    j_j___libc_free_0(v22);
}
