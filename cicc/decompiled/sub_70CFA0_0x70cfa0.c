// Function: sub_70CFA0
// Address: 0x70cfa0
//
_QWORD *__fastcall sub_70CFA0(__int64 a1, __int64 a2, __int64 *a3, _BOOL4 *a4)
{
  int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-50h]
  __int16 v12[32]; // [rsp+10h] [rbp-40h] BYREF

  while ( 1 )
  {
    sub_620DE0(v12, *(_QWORD *)(a2 + 128));
    v6 = sub_620E90(a1);
    sub_621340((unsigned __int16 *)(a1 + 176), v6, v12, 0, a4);
    result = *(_QWORD **)(*(_QWORD *)(a2 + 40) + 32LL);
    v8 = result[21];
    if ( a3 )
    {
      v10 = result[21];
      result = (_QWORD *)sub_724980();
      v9 = *a3;
      v8 = v10;
      result[2] = a2;
      *result = v9;
      *a3 = (__int64)result;
    }
    if ( *(_BYTE *)(v8 + 113) != 2 )
      break;
    a2 = *(_QWORD *)(v8 + 120);
  }
  return result;
}
