// Function: sub_39F14C0
// Address: 0x39f14c0
//
_BYTE *__fastcall sub_39F14C0(unsigned int *a1, __int64 a2)
{
  _QWORD *v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rdi
  _BYTE *v9; // rax
  _BYTE *result; // rax

  v3 = *(_QWORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 7u )
  {
    v4 = sub_16E7EE0(a2, "<MCInst ", 8u);
  }
  else
  {
    v4 = a2;
    *v3 = 0x2074736E49434D3CLL;
    *(_QWORD *)(a2 + 24) += 8LL;
  }
  sub_16E7A90(v4, *a1);
  v5 = a1[6];
  if ( (_DWORD)v5 )
  {
    v6 = 16 * v5;
    v7 = 0;
    do
    {
      v9 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v9 )
      {
        sub_16E7EE0(a2, " ", 1u);
      }
      else
      {
        *v9 = 32;
        ++*(_QWORD *)(a2 + 24);
      }
      v8 = v7 + *((_QWORD *)a1 + 2);
      v7 += 16;
      sub_39F15E0(v8, a2);
    }
    while ( v6 != v7 );
  }
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a2, ">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
