// Function: sub_3950160
// Address: 0x3950160
//
__int64 __fastcall sub_3950160(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  const char *v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r15
  const char *v9; // r14
  bool v10; // r8
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax

  v5 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v5 + 16) )
    v5 = 0;
  v6 = sub_1649960(v5);
  v8 = v7;
  v9 = v6;
  v10 = sub_394FE80((__int64)a1, a2, 3u, 2u, 0);
  result = 0;
  if ( v10 )
  {
    v12 = v8;
    if ( v8 > 1 )
    {
      result = v8 - 2;
      v12 = 2;
      if ( v8 - 2 > 6 )
        result = 7;
    }
    v13 = result;
    v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    return sub_1AB2190(
             *(_QWORD *)(a2 - 24 * v14),
             *(_QWORD *)(a2 + 24 * (1 - v14)),
             *(_QWORD **)(a2 + 24 * (2 - v14)),
             a3,
             *a1,
             *(_QWORD *)(a2 - 24 * v14),
             (__int64)&v9[v12],
             v13);
  }
  return result;
}
