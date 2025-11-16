// Function: sub_359C720
// Address: 0x359c720
//
__int64 __fastcall sub_359C720(__int64 a1, unsigned int a2, unsigned int a3, int a4, int a5, __int64 a6, __int64 a7)
{
  int v11; // eax
  int v12; // r8d
  int v13; // r9d
  unsigned __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  int v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v16[0] = a4;
  if ( a2 <= a3 )
    return 0;
  v14 = sub_2EBEE10(*(_QWORD *)(a1 + 24), v16[0]);
  if ( a3 == a5 )
  {
    v15 = a6 + 32LL * (a2 - 1);
    if ( sub_359BBF0(v15, v16) )
      return (unsigned int)*sub_2FFAE70(v15, v16);
  }
  if ( sub_359BBF0(a6 + 32LL * a2, v16) )
    return (unsigned int)*sub_2FFAE70(a6 + 32LL * a2, v16);
  if ( *(_WORD *)(v14 + 68) && *(_WORD *)(v14 + 68) != 68 || a7 != *(_QWORD *)(v14 + 24) )
    return (unsigned int)v16[0];
  if ( a3 + 1 == a2 )
    return sub_3598140(v14, a7);
  if ( a3 + 1 >= a2 )
    return 0;
  v11 = sub_3598190(v14, a7);
  return sub_359C720(a1, a2 - 1, a3, v11, v12, v13, a7);
}
