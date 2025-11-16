// Function: sub_2F5DFB0
// Address: 0x2f5dfb0
//
__int64 __fastcall sub_2F5DFB0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r14
  unsigned int v9; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned int v13; // [rsp+4h] [rbp-6Ch]
  unsigned int v15; // [rsp+1Ch] [rbp-54h] BYREF
  _BYTE v16[80]; // [rsp+20h] [rbp-50h] BYREF

  v13 = a6 + 1;
LABEL_2:
  if ( *a2 == a2[1] )
    return 1;
  while ( 1 )
  {
    v8 = sub_2F505C0(a1, (__int64)a2);
    v9 = sub_2F5D4A0(a1, v8, a3, a4, a5, v13);
    if ( v9 == -1 )
      return 0;
    if ( !v9 )
    {
      if ( !*(_DWORD *)(v8 + 8) )
        goto LABEL_2;
      return 0;
    }
    sub_2E20EE0(*(_QWORD **)(a1 + 40), v8, v9);
    v15 = *(_DWORD *)(v8 + 112);
    sub_2F58C70((__int64)v16, a4, &v15, v11, v12);
    if ( *a2 == a2[1] )
      return 1;
  }
}
