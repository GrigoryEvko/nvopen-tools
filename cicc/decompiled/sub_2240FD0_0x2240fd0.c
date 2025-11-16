// Function: sub_2240FD0
// Address: 0x2240fd0
//
unsigned __int64 *__fastcall sub_2240FD0(unsigned __int64 *a1, size_t a2, __int64 a3, size_t a4, char a5)
{
  unsigned __int64 v6; // rcx
  unsigned __int64 v8; // rdi
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rax
  size_t v13; // rcx
  size_t v14; // rdi
  _BYTE *v15; // rsi
  _BYTE *v16; // rdi
  _BYTE *v17; // rdi

  v6 = a1[1];
  if ( a4 > a3 + 0x3FFFFFFFFFFFFFFFLL - v6 )
    sub_4262D8((__int64)"basic_string::_M_replace_aux");
  v8 = *a1;
  v11 = v6 + a4 - a3;
  if ( (unsigned __int64 *)v8 == a1 + 2 )
    v12 = 15;
  else
    v12 = a1[2];
  if ( v12 < v11 )
  {
    sub_2240BB0(a1, a2, a3, 0, a4);
    v8 = *a1;
    if ( !a4 )
      goto LABEL_13;
  }
  else
  {
    v13 = v6 - (a3 + a2);
    if ( !v13 || a3 == a4 )
    {
LABEL_9:
      if ( !a4 )
        goto LABEL_13;
      goto LABEL_10;
    }
    v14 = a2 + v8;
    v15 = (_BYTE *)(v14 + a3);
    v16 = (_BYTE *)(a4 + v14);
    if ( v13 != 1 )
    {
      memmove(v16, v15, v13);
      v8 = *a1;
      goto LABEL_9;
    }
    *v16 = *v15;
    v8 = *a1;
    if ( !a4 )
      goto LABEL_13;
  }
LABEL_10:
  v17 = (_BYTE *)(a2 + v8);
  if ( a4 == 1 )
    *v17 = a5;
  else
    memset(v17, a5, a4);
  v8 = *a1;
LABEL_13:
  a1[1] = v11;
  *(_BYTE *)(v8 + v11) = 0;
  return a1;
}
