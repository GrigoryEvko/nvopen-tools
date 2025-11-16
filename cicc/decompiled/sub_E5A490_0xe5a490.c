// Function: sub_E5A490
// Address: 0xe5a490
//
__int64 __fastcall sub_E5A490(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        const void *a12,
        size_t a13)
{
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi

  v16 = sub_904010(*(_QWORD *)(a1 + 304), "\t.loc\t");
  v17 = sub_CB59D0(v16, a2);
  v18 = sub_904010(v17, " ");
  v19 = sub_CB59D0(v18, a3);
  v20 = sub_904010(v19, " ");
  sub_CB59D0(v20, a4);
  v21 = sub_904010(*(_QWORD *)(a1 + 304), ", function_name ");
  sub_EA12C0(a8, v21, 0);
  v22 = sub_904010(*(_QWORD *)(a1 + 304), ", inlined_at ");
  v23 = sub_CB59D0(v22, a5);
  v24 = sub_904010(v23, " ");
  v25 = sub_CB59D0(v24, a6);
  v26 = sub_904010(v25, " ");
  sub_CB59D0(v26, a7);
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 312) + 354LL) )
    goto LABEL_10;
  if ( (a9 & 2) == 0 )
  {
    if ( (a9 & 4) == 0 )
      goto LABEL_4;
LABEL_20:
    sub_904010(*(_QWORD *)(a1 + 304), " prologue_end");
    if ( (a9 & 8) == 0 )
      goto LABEL_5;
LABEL_21:
    sub_904010(*(_QWORD *)(a1 + 304), " epilogue_begin");
    goto LABEL_5;
  }
  sub_904010(*(_QWORD *)(a1 + 304), " basic_block");
  if ( (a9 & 4) != 0 )
    goto LABEL_20;
LABEL_4:
  if ( (a9 & 8) != 0 )
    goto LABEL_21;
LABEL_5:
  if ( (((unsigned __int8)a9 ^ *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1786LL)) & 1) != 0 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), " is_stmt ");
    v36 = *(_QWORD *)(a1 + 304);
    if ( (a9 & 1) != 0 )
      sub_904010(v36, "1");
    else
      sub_904010(v36, "0");
  }
  if ( (_DWORD)a10 )
  {
    v35 = sub_904010(*(_QWORD *)(a1 + 304), " isa ");
    sub_CB59D0(v35, (unsigned int)a10);
  }
  if ( (_DWORD)a11 )
  {
    v34 = sub_904010(*(_QWORD *)(a1 + 304), " discriminator ");
    sub_CB59D0(v34, (unsigned int)a11);
  }
LABEL_10:
  if ( *(_BYTE *)(a1 + 745) )
  {
    sub_C66A60(*(_QWORD *)(a1 + 304), *(_DWORD *)(*(_QWORD *)(a1 + 312) + 396LL));
    v28 = sub_A51340(
            *(_QWORD *)(a1 + 304),
            *(const void **)(*(_QWORD *)(a1 + 312) + 48LL),
            *(_QWORD *)(*(_QWORD *)(a1 + 312) + 56LL));
    v29 = *(_BYTE **)(v28 + 32);
    if ( (unsigned __int64)v29 >= *(_QWORD *)(v28 + 24) )
    {
      v28 = sub_CB5D20(v28, 32);
    }
    else
    {
      *(_QWORD *)(v28 + 32) = v29 + 1;
      *v29 = 32;
    }
    v30 = sub_A51340(v28, a12, a13);
    v31 = *(_BYTE **)(v30 + 32);
    if ( (unsigned __int64)v31 >= *(_QWORD *)(v30 + 24) )
    {
      v30 = sub_CB5D20(v30, 58);
    }
    else
    {
      *(_QWORD *)(v30 + 32) = v31 + 1;
      *v31 = 58;
    }
    v32 = sub_CB59D0(v30, a3);
    v33 = *(_BYTE **)(v32 + 32);
    if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
    {
      v32 = sub_CB5D20(v32, 58);
    }
    else
    {
      *(_QWORD *)(v32 + 32) = v33 + 1;
      *v33 = 58;
    }
    sub_CB59D0(v32, a4);
  }
  sub_E4D880(a1);
  return sub_E97590(a1, a2, a3, a4, a9, a10, a11);
}
