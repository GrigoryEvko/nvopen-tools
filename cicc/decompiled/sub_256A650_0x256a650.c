// Function: sub_256A650
// Address: 0x256a650
//
__int64 __fastcall sub_256A650(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int64 i; // rbx
  _WORD *v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // r8
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 v15; // bl
  unsigned __int8 v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+10h] [rbp-80h]
  __int64 v19; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v20; // [rsp+20h] [rbp-70h] BYREF
  size_t v21; // [rsp+28h] [rbp-68h]
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int8 *v23[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+50h] [rbp-40h] BYREF

  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 16LL))(a3);
  v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 24LL))(a3);
  if ( *(_BYTE *)(a1 + 16) )
    sub_904010(a2, "</tr><tr>");
  if ( v4 == v19 )
    return 0;
  v17 = 0;
  for ( i = 0; i != 64; ++i )
  {
    v20 = (unsigned __int8 *)v22;
    sub_2539970((__int64 *)&v20, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v21 )
    {
      if ( *(_BYTE *)(a1 + 16) )
      {
        v10 = sub_904010(a2, "<td colspan=\"1\" port=\"s");
        v11 = sub_CB59D0(v10, i);
        v12 = sub_904010(v11, "\">");
        v13 = sub_CB6200(v12, v20, v21);
        sub_904010(v13, "</td>");
      }
      else
      {
        if ( i )
          sub_904010(a2, "|");
        v6 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 1u )
        {
          v7 = sub_CB6200(a2, "<s", 2u);
        }
        else
        {
          v7 = a2;
          *v6 = 29500;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        v8 = sub_CB59D0(v7, i);
        v9 = *(_BYTE **)(v8 + 32);
        if ( *(_BYTE **)(v8 + 24) == v9 )
        {
          v8 = sub_CB6200(v8, (unsigned __int8 *)">", 1u);
        }
        else
        {
          *v9 = 62;
          ++*(_QWORD *)(v8 + 32);
        }
        v18 = v8;
        sub_C67200((__int64 *)v23, (__int64)&v20);
        sub_CB6200(v18, v23[0], (size_t)v23[1]);
        if ( (__int64 *)v23[0] != &v24 )
          j_j___libc_free_0((unsigned __int64)v23[0]);
      }
      if ( v20 != (unsigned __int8 *)v22 )
        j_j___libc_free_0((unsigned __int64)v20);
      v17 = 1;
LABEL_16:
      v4 += 8;
      if ( v19 == v4 )
        return v17;
      continue;
    }
    if ( v20 == (unsigned __int8 *)v22 )
      goto LABEL_16;
    v4 += 8;
    j_j___libc_free_0((unsigned __int64)v20);
    if ( v19 == v4 )
      return v17;
  }
  if ( v17 )
  {
    v15 = *(_BYTE *)(a1 + 16);
    if ( v15 )
    {
      sub_904010(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>");
      return v15;
    }
    else
    {
      sub_904010(a2, "|<s64>truncated...");
    }
  }
  return v17;
}
