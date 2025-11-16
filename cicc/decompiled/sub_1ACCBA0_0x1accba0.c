// Function: sub_1ACCBA0
// Address: 0x1accba0
//
__int64 __fastcall sub_1ACCBA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  unsigned __int8 v6; // cl
  __int64 result; // rax
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
  v5 = *(_BYTE *)(a2 + 16);
  v6 = *(_BYTE *)(a3 + 16);
  if ( v5 <= 0x10u )
  {
    if ( v6 <= 0x10u )
    {
      if ( a2 == a3 )
        return 0;
      else
        return sub_1ACC630((__int64 *)a1, a2, a3);
    }
    return 1;
  }
  result = 0xFFFFFFFFLL;
  if ( v6 <= 0x10u )
    return result;
  if ( v5 == 20 )
  {
    if ( v6 == 20 )
      return sub_1ACB4F0((__int64 *)a1, a2, a3);
    return 1;
  }
  if ( v6 != 20 )
  {
    v8 = *(_DWORD *)(a1 + 32);
    v14 = a2;
    v15 = v8;
    sub_1ACBAF0((__int64)v12, a1 + 16, &v14, &v15);
    v9 = *(_DWORD *)(a1 + 64);
    v10 = a3;
    v11 = v9;
    sub_1ACBAF0((__int64)&v14, a1 + 48, &v10, &v11);
    return sub_1ACA9E0(a1, *(int *)(v13 + 8), *(int *)(v16 + 8));
  }
  return result;
}
