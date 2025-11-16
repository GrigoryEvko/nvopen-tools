// Function: sub_38C4070
// Address: 0x38c4070
//
bool __fastcall sub_38C4070(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  bool result; // al
  unsigned int v11; // [rsp+4h] [rbp-1Ch] BYREF
  unsigned int *v12; // [rsp+8h] [rbp-18h] BYREF

  v3 = a2;
  v5 = a1 + 984;
  v6 = v5;
  v7 = *(_QWORD *)(a1 + 992);
  v11 = a3;
  if ( v7 )
  {
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 + 16);
        v9 = *(_QWORD *)(v7 + 24);
        if ( a3 <= *(_DWORD *)(v7 + 32) )
          break;
        v7 = *(_QWORD *)(v7 + 24);
        if ( !v9 )
          goto LABEL_6;
      }
      v6 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v8 );
LABEL_6:
    if ( v5 != v6 && a3 >= *(_DWORD *)(v6 + 32) )
    {
      result = 0;
      if ( !(_DWORD)v3 )
        goto LABEL_9;
LABEL_13:
      if ( (unsigned int)v3 < *(_DWORD *)(v6 + 168) )
        return *(_QWORD *)(*(_QWORD *)(v6 + 160) + 72 * v3 + 8) != 0;
      return result;
    }
  }
  v12 = &v11;
  v6 = sub_38C3E00((_QWORD *)(a1 + 976), v6, &v12);
  result = 0;
  if ( (_DWORD)v3 )
    goto LABEL_13;
LABEL_9:
  if ( *(_WORD *)(a1 + 1160) > 4u )
    return *(_QWORD *)(v6 + 464) != 0;
  return result;
}
