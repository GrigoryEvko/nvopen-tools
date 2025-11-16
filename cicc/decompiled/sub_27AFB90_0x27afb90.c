// Function: sub_27AFB90
// Address: 0x27afb90
//
__int64 __fastcall sub_27AFB90(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 i; // rbx
  __int64 v8; // rdi
  __int64 v9; // rax
  _DWORD *v10; // rdx
  _WORD *v11; // rdx
  _BYTE *v12; // rax

  if ( a3 )
    sub_904010(a2, "ExpressionTypeBasic, ");
  v4 = sub_904010(a2, "opcode = ");
  v5 = sub_CB59D0(v4, *(unsigned int *)(a1 + 12));
  sub_904010(v5, ", ");
  sub_904010(a2, "operands = {");
  v6 = *(unsigned int *)(a1 + 36);
  if ( (_DWORD)v6 )
  {
    for ( i = 0; i != v6; ++i )
    {
      while ( 1 )
      {
        v12 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v12 )
        {
          v8 = sub_CB6200(a2, (unsigned __int8 *)"[", 1u);
        }
        else
        {
          *v12 = 91;
          v8 = a2;
          ++*(_QWORD *)(a2 + 32);
        }
        v9 = sub_CB59D0(v8, i);
        v10 = *(_DWORD **)(v9 + 32);
        if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 3u )
        {
          sub_CB6200(v9, "] = ", 4u);
        }
        else
        {
          *v10 = 540876893;
          *(_QWORD *)(v9 + 32) += 4LL;
        }
        sub_A5BF40(*(unsigned __int8 **)(*(_QWORD *)(a1 + 24) + 8 * i), a2, 1, 0);
        v11 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 1u )
          break;
        ++i;
        *v11 = 8224;
        *(_QWORD *)(a2 + 32) += 2LL;
        if ( v6 == i )
          return sub_904010(a2, "} ");
      }
      sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
    }
  }
  return sub_904010(a2, "} ");
}
