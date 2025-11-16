// Function: sub_E51E80
// Address: 0xe51e80
//
_BYTE *__fastcall sub_E51E80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        char a6,
        char a7,
        const void *a8,
        size_t a9,
        __int64 a10)
{
  unsigned int v11; // ebx
  _BYTE *result; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
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

  v11 = a3;
  result = (_BYTE *)sub_E99480(a1, a2, a3, a10);
  if ( (_BYTE)result )
  {
    v13 = sub_904010(*(_QWORD *)(a1 + 304), "\t.cv_loc\t");
    v14 = sub_CB59D0(v13, (unsigned int)a2);
    v15 = sub_904010(v14, " ");
    v16 = sub_CB59D0(v15, v11);
    v17 = sub_904010(v16, " ");
    v18 = sub_CB59D0(v17, a4);
    v19 = sub_904010(v18, " ");
    sub_CB59D0(v19, a5);
    if ( a6 )
    {
      sub_904010(*(_QWORD *)(a1 + 304), " prologue_end");
      if ( !a7 )
      {
LABEL_4:
        if ( !*(_BYTE *)(a1 + 745) )
          return sub_E4D880(a1);
LABEL_8:
        sub_C66A60(*(_QWORD *)(a1 + 304), *(_DWORD *)(*(_QWORD *)(a1 + 312) + 396LL));
        v20 = sub_A51340(
                *(_QWORD *)(a1 + 304),
                *(const void **)(*(_QWORD *)(a1 + 312) + 48LL),
                *(_QWORD *)(*(_QWORD *)(a1 + 312) + 56LL));
        v21 = sub_A51310(v20, 0x20u);
        v22 = sub_A51340(v21, a8, a9);
        v23 = sub_A51310(v22, 0x3Au);
        v24 = sub_CB59D0(v23, a4);
        v25 = sub_A51310(v24, 0x3Au);
        sub_CB59D0(v25, a5);
        return sub_E4D880(a1);
      }
    }
    else if ( !a7 )
    {
      goto LABEL_4;
    }
    sub_904010(*(_QWORD *)(a1 + 304), " is_stmt 1");
    if ( !*(_BYTE *)(a1 + 745) )
      return sub_E4D880(a1);
    goto LABEL_8;
  }
  return result;
}
