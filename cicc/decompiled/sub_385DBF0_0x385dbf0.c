// Function: sub_385DBF0
// Address: 0x385dbf0
//
char __fastcall sub_385DBF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rbx
  __int64 v6; // rbx
  __int64 v7; // r15
  char result; // al
  __int64 v9; // [rsp-48h] [rbp-48h]
  __int64 v10; // [rsp-40h] [rbp-40h]

  v3 = *(unsigned int *)(a2 + 32);
  if ( !(_DWORD)v3 )
    return 0;
  v9 = 4 * v3;
  v10 = 0;
  while ( 1 )
  {
    v5 = *(unsigned int *)(a3 + 32);
    if ( (_DWORD)v5 )
      break;
LABEL_8:
    v10 += 4;
    if ( v9 == v10 )
      return 0;
  }
  v6 = 4 * v5;
  v7 = 0;
  while ( 1 )
  {
    result = sub_385DBB0(a1, *(_DWORD *)(*(_QWORD *)(a2 + 24) + v10), *(_DWORD *)(*(_QWORD *)(a3 + 24) + v7));
    if ( result )
      return result;
    v7 += 4;
    if ( v6 == v7 )
      goto LABEL_8;
  }
}
