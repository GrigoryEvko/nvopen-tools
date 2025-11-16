// Function: sub_1DF8250
// Address: 0x1df8250
//
_BYTE *__fastcall sub_1DF8250(int *a1, __int64 a2)
{
  __int64 v3; // rdi
  _DWORD *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdi
  _BYTE *result; // rax

  v3 = a2;
  v4 = *(_DWORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 3u )
  {
    v3 = sub_16E7EE0(a2, "[R: ", 4u);
  }
  else
  {
    *v4 = 540693083;
    *(_QWORD *)(a2 + 24) += 4LL;
  }
  v5 = sub_16E7AB0(v3, *a1);
  v6 = *(_QWORD *)(v5 + 24);
  v7 = v5;
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 4 )
  {
    v7 = sub_16E7EE0(v5, ", P: ", 5u);
  }
  else
  {
    *(_DWORD *)v6 = 978329644;
    *(_BYTE *)(v6 + 4) = 32;
    *(_QWORD *)(v5 + 24) += 5LL;
  }
  v8 = sub_16E7AB0(v7, a1[1]);
  result = *(_BYTE **)(v8 + 24);
  if ( *(_BYTE **)(v8 + 16) == result )
    return (_BYTE *)sub_16E7EE0(v8, "]", 1u);
  *result = 93;
  ++*(_QWORD *)(v8 + 24);
  return result;
}
