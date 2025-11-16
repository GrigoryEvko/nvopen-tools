// Function: sub_1061FD0
// Address: 0x1061fd0
//
__int64 __fastcall sub_1061FD0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  int v4; // r12d
  __int64 v5; // r15
  int v6; // r12d
  __int64 *v7; // r14
  int v9; // r14d
  __int64 v10; // [rsp+0h] [rbp-60h]
  int v11; // [rsp+8h] [rbp-58h]
  unsigned int i; // [rsp+Ch] [rbp-54h]
  _BYTE v13[80]; // [rsp+10h] [rbp-50h] BYREF

  sub_1061A50((__int64)v13, a2, a3, a4);
  v4 = *(_DWORD *)(a1 + 56);
  v5 = *(_QWORD *)(a1 + 40);
  if ( !v4 )
    return 0;
  v6 = v4 - 1;
  v10 = sub_1061AC0();
  v11 = 1;
  for ( i = v6 & sub_1061E10((__int64)v13); ; i = v9 )
  {
    v7 = (__int64 *)(v5 + 8LL * i);
    if ( sub_1061AE0((__int64)v13, *v7) )
      break;
    if ( sub_1061B40(*v7, v10) )
      return 0;
    v9 = v6 & (v11 + i);
    ++v11;
  }
  if ( v7 == (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 56)) )
    return 0;
  else
    return *v7;
}
