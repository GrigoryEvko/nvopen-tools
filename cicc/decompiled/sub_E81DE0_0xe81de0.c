// Function: sub_E81DE0
// Address: 0xe81de0
//
_BYTE *__fastcall sub_E81DE0(unsigned int *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rdi
  _BYTE *v11; // rax
  _BYTE *result; // rax

  v5 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 7u )
  {
    v6 = sub_CB6200(a2, "<MCInst ", 8u);
  }
  else
  {
    v6 = a2;
    *v5 = 0x2074736E49434D3CLL;
    *(_QWORD *)(a2 + 32) += 8LL;
  }
  sub_CB59D0(v6, *a1);
  v7 = a1[6];
  if ( (_DWORD)v7 )
  {
    v8 = 16 * v7;
    v9 = 0;
    do
    {
      v11 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v11 )
      {
        sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v11 = 32;
        ++*(_QWORD *)(a2 + 32);
      }
      v10 = v9 + *((_QWORD *)a1 + 2);
      v9 += 16;
      sub_E81F00(v10, a2, a3);
    }
    while ( v8 != v9 );
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
