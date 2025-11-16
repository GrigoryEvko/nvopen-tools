// Function: sub_29DA390
// Address: 0x29da390
//
__int64 __fastcall sub_29DA390(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl
  int v8; // eax
  int v9; // eax
  __int64 v10; // [rsp+0h] [rbp-90h] BYREF
  int v11; // [rsp+8h] [rbp-88h] BYREF
  _BYTE v12[16]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v13; // [rsp+20h] [rbp-70h]
  __int64 v14; // [rsp+40h] [rbp-50h] BYREF
  int v15; // [rsp+48h] [rbp-48h] BYREF
  __int64 v16; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == a2 )
    return (unsigned int)-(a3 != v4);
  if ( a3 == v4 )
    return 1;
  v5 = *(_BYTE *)a2;
  v6 = *(_BYTE *)a3;
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( v6 <= 0x15u )
    {
      if ( a2 != a3 )
        return sub_29D9730((__int64 *)a1, (unsigned __int8 *)a2, (unsigned __int8 *)a3);
      return 0;
    }
    return 1;
  }
  if ( v6 <= 0x15u )
    return 0xFFFFFFFFLL;
  if ( v5 == 24 )
  {
    if ( v6 != 24 )
      return 1;
    if ( a2 == a3 )
      return 0;
    return sub_29D9F90((__int64 *)a1, *(_QWORD *)(a2 + 24), *(_QWORD *)(a3 + 24));
  }
  else
  {
    if ( v6 == 24 )
      return 0xFFFFFFFFLL;
    if ( v5 == 25 )
    {
      if ( v6 != 25 )
        return 1;
      return sub_29D88C0((__int64 *)a1, a2, a3);
    }
    else
    {
      if ( v6 == 25 )
        return 0xFFFFFFFFLL;
      v8 = *(_DWORD *)(a1 + 32);
      v14 = a2;
      v15 = v8;
      sub_29D8B70((__int64)v12, a1 + 16, &v14, &v15);
      v9 = *(_DWORD *)(a1 + 64);
      v10 = a3;
      v11 = v9;
      sub_29D8B70((__int64)&v14, a1 + 48, &v10, &v11);
      return sub_29D7CF0(a1, *(int *)(v13 + 8), *(int *)(v16 + 8));
    }
  }
}
