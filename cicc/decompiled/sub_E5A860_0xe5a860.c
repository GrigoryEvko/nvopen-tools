// Function: sub_E5A860
// Address: 0xe5a860
//
__int64 __fastcall sub_E5A860(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        int a5,
        unsigned int a6,
        unsigned int a7,
        const void *a8,
        size_t a9,
        const void *a10,
        size_t a11)
{
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
  {
    sub_E7BC40(a1, *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL));
    return sub_E97590(a1, a2, a3, a4, a5, a6, a7);
  }
  v15 = sub_904010(*(_QWORD *)(a1 + 304), "\t.loc\t");
  v16 = sub_CB59D0(v15, a2);
  v17 = sub_904010(v16, " ");
  v18 = sub_CB59D0(v17, a3);
  v19 = sub_904010(v18, " ");
  sub_CB59D0(v19, a4);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 354LL) )
  {
    if ( (a5 & 2) != 0 )
      sub_904010(*(_QWORD *)(a1 + 304), " basic_block");
    if ( (a5 & 4) != 0 )
      sub_904010(*(_QWORD *)(a1 + 304), " prologue_end");
    if ( (a5 & 8) != 0 )
      sub_904010(*(_QWORD *)(a1 + 304), " epilogue_begin");
    if ( (((unsigned __int8)a5 ^ *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1786LL)) & 1) != 0 )
    {
      sub_904010(*(_QWORD *)(a1 + 304), " is_stmt ");
      v28 = *(_QWORD *)(a1 + 304);
      if ( (a5 & 1) != 0 )
        sub_904010(v28, "1");
      else
        sub_904010(v28, "0");
    }
    if ( a6 )
    {
      v27 = sub_904010(*(_QWORD *)(a1 + 304), " isa ");
      sub_CB59D0(v27, a6);
    }
    if ( a7 )
    {
      v26 = sub_904010(*(_QWORD *)(a1 + 304), " discriminator ");
      sub_CB59D0(v26, a7);
      if ( !*(_BYTE *)(a1 + 745) )
        goto LABEL_14;
LABEL_16:
      sub_C66A60(*(_QWORD *)(a1 + 304), *(_DWORD *)(*(_QWORD *)(a1 + 312) + 396LL));
      v21 = sub_A51340(
              *(_QWORD *)(a1 + 304),
              *(const void **)(*(_QWORD *)(a1 + 312) + 48LL),
              *(_QWORD *)(*(_QWORD *)(a1 + 312) + 56LL));
      sub_A51310(v21, 0x20u);
      if ( a11 )
      {
        sub_A51340(*(_QWORD *)(a1 + 304), a10, a11);
      }
      else
      {
        v22 = sub_A51340(*(_QWORD *)(a1 + 304), a8, a9);
        v23 = sub_A51310(v22, 0x3Au);
        v24 = sub_CB59D0(v23, a3);
        v25 = sub_A51310(v24, 0x3Au);
        sub_CB59D0(v25, a4);
      }
      goto LABEL_14;
    }
  }
  if ( *(_BYTE *)(a1 + 745) )
    goto LABEL_16;
LABEL_14:
  sub_E4D880(a1);
  return sub_E97590(a1, a2, a3, a4, a5, a6, a7);
}
