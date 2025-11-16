// Function: sub_1C73930
// Address: 0x1c73930
//
__int64 __fastcall sub_1C73930(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10,
        _QWORD *a11)
{
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // esi
  __int64 v18; // rax
  unsigned int v19; // r12d
  int v21; // edx
  unsigned __int64 v22; // rax
  __int64 v23; // rbx
  int v24; // ecx
  int v25; // edx
  unsigned int v26; // eax

  v15 = sub_157EBA0(a2);
  v16 = *(_QWORD *)(v15 - 72);
  v17 = *(_WORD *)(v16 + 18) & 0x7FFF;
  if ( ((v17 - 36) & 0xFFFFFFFB) == 0 )
  {
    if ( a3 != *(_QWORD *)(v15 - 24) )
    {
LABEL_3:
      v18 = *a10;
      goto LABEL_4;
    }
LABEL_11:
    *a10 = *(_QWORD *)(v16 - 48);
    *a11 = *(_QWORD *)(v16 - 24);
    goto LABEL_3;
  }
  v21 = *(_WORD *)(v16 + 18) & 0x7FFB;
  if ( v21 == 34 )
  {
    if ( *(_QWORD *)(v15 - 24) != a3 )
      goto LABEL_3;
  }
  else
  {
    if ( v21 == 35 )
    {
      if ( a7 != *(_QWORD *)(v15 - 24) )
        goto LABEL_3;
      goto LABEL_11;
    }
    if ( ((v17 - 37) & 0xFFFFFFFB) != 0 )
      return 0;
    if ( a7 != *(_QWORD *)(v15 - 24) )
      goto LABEL_3;
  }
  *a11 = *(_QWORD *)(v16 - 48);
  v18 = *(_QWORD *)(v16 - 24);
  *a10 = v18;
LABEL_4:
  if ( !v18 || *a11 == 0 || *a11 != a9 || a8 != v18 )
    return 0;
  v22 = sub_157EBA0(a5);
  v23 = *(_QWORD *)(v22 - 72);
  v24 = *(unsigned __int16 *)(v23 + 18);
  BYTE1(v24) &= ~0x80u;
  if ( ((v24 - 36) & 0xFFFFFFFB) == 0 )
  {
    if ( a4 != *(_QWORD *)(v22 - 24) )
      return 1;
LABEL_27:
    if ( !sub_13FC1A0(a1, *(_QWORD *)(v23 - 48)) )
      sub_13FC1A0(a1, *(_QWORD *)(v23 - 24));
    return 1;
  }
  v25 = *(_WORD *)(v23 + 18) & 0x7FFB;
  if ( v25 == 34 )
  {
    if ( a4 != *(_QWORD *)(v22 - 24) )
      return 1;
LABEL_23:
    LOBYTE(v26) = sub_13FC1A0(a1, *(_QWORD *)(v23 - 48));
    v19 = v26;
    if ( (_BYTE)v26 )
    {
      sub_13FC1A0(a1, *(_QWORD *)(v23 - 24));
      return v19;
    }
    return 1;
  }
  if ( v25 == 35 )
  {
    if ( a6 != *(_QWORD *)(v22 - 24) )
      return 1;
    goto LABEL_27;
  }
  if ( ((v24 - 37) & 0xFFFFFFFB) == 0 )
  {
    if ( a6 != *(_QWORD *)(v22 - 24) )
      return 1;
    goto LABEL_23;
  }
  return 0;
}
