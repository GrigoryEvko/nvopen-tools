// Function: sub_2F76D20
// Address: 0x2f76d20
//
__int64 __fastcall sub_2F76D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // [rsp-10h] [rbp-30h]

  result = 3 * a3;
  v7 = a2 + 24 * a3;
  if ( a2 != v7 )
  {
    v8 = a2;
    do
    {
      v11 = *(_QWORD *)(v8 + 16);
      v8 += 24;
      v9 = sub_2F74C60(a1 + 96, a2, a3, a4, a5, a6, *(_QWORD *)(v8 - 24), *(_QWORD *)(v8 - 16), v11);
      a2 = *(unsigned int *)(v8 - 24);
      result = sub_2F74DB0(a1, a2, v9, v10, v9 | *(_QWORD *)(v8 - 16), v10 | *(_QWORD *)(v8 - 8));
    }
    while ( v7 != v8 );
  }
  return result;
}
