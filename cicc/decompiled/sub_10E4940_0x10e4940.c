// Function: sub_10E4940
// Address: 0x10e4940
//
char __fastcall sub_10E4940(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char result; // al
  __int64 v7; // rdx
  _QWORD *v8; // [rsp+8h] [rbp-28h] BYREF

  v8 = *(_QWORD **)(a1 + 8);
  result = sub_995E90(&v8, a2, a3, a4, a5);
  if ( !result || **(_QWORD **)(a1 + 8) != a3 )
    return 0;
  v7 = *(_QWORD *)(a2 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) )
    return ((*(_DWORD *)a1 - 246) & 0xFFFFFFFD) != 0;
  return result;
}
