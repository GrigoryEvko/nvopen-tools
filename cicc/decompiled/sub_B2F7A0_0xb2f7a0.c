// Function: sub_B2F7A0
// Address: 0xb2f7a0
//
_QWORD *__fastcall sub_B2F7A0(_QWORD *a1, _BYTE *a2, unsigned __int64 a3, int a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // rcx
  _BYTE *v7; // r14
  unsigned __int64 v8; // r13
  _QWORD *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax

  v6 = (unsigned int)(a4 - 7);
  v7 = a2;
  v8 = a3;
  v9 = a1 + 2;
  if ( a3 )
  {
    if ( *a2 == 1 )
    {
      v8 = a3 - 1;
      v7 = a2 + 1;
    }
    *a1 = v9;
    a1[1] = 0;
    *((_BYTE *)a1 + 16) = 0;
    if ( (unsigned int)v6 > 1 )
    {
      v15 = 0x3FFFFFFFFFFFFFFFLL;
      goto LABEL_13;
    }
    if ( a6 )
      goto LABEL_6;
  }
  else
  {
    *a1 = v9;
    a1[1] = 0;
    *((_BYTE *)a1 + 16) = 0;
    if ( (unsigned int)v6 > 1 )
      goto LABEL_14;
    if ( a6 )
    {
LABEL_6:
      if ( a6 > 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_20;
      sub_2241490(a1, a5, a6, v6);
      goto LABEL_8;
    }
  }
  sub_2241490(a1, "<unknown>", 9, v6);
LABEL_8:
  v10 = a1[1];
  v11 = *a1;
  v12 = v10 + 1;
  if ( v9 == (_QWORD *)*a1 )
    v13 = 15;
  else
    v13 = a1[2];
  if ( v12 > v13 )
  {
    sub_2240BB0(a1, a1[1], 0, 0, 1);
    v11 = *a1;
    v12 = v10 + 1;
  }
  *(_BYTE *)(v11 + v10) = 59;
  v14 = *a1;
  a1[1] = v12;
  *(_BYTE *)(v14 + v10 + 1) = 0;
  v15 = 0x3FFFFFFFFFFFFFFFLL - a1[1];
LABEL_13:
  if ( v8 > v15 )
LABEL_20:
    sub_4262D8((__int64)"basic_string::append");
LABEL_14:
  sub_2241490(a1, v7, v8, v6);
  return a1;
}
