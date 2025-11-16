// Function: sub_2C4CBB0
// Address: 0x2c4cbb0
//
__int64 __fastcall sub_2C4CBB0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r15
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v13; // [rsp+0h] [rbp-50h]
  int v15; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_QWORD *)(a3 - 32);
  if ( *(_BYTE *)v5 != 17 )
    v5 = 0;
  if ( *(_BYTE *)v6 != 17 )
    v6 = 0;
  v7 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  if ( (_DWORD)v7 == (_DWORD)v8 )
    return 0;
  v9 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  v15 = v10;
  v13 = v9;
  v11 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  if ( v15 )
    return 0;
  if ( v13 > v11 || (_DWORD)v7 != a4 && v13 >= v11 && ((unsigned int)v7 > (unsigned int)v8 || (_DWORD)v8 == a4) )
    return a2;
  return a3;
}
