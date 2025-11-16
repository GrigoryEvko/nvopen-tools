// Function: sub_B49A80
// Address: 0xb49a80
//
bool __fastcall sub_B49A80(__int64 a1)
{
  bool v1; // sf
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v9; // [rsp+8h] [rbp-68h]
  int v10; // [rsp+1Ch] [rbp-54h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-50h] BYREF
  int v12; // [rsp+30h] [rbp-40h]
  _BYTE v13[60]; // [rsp+34h] [rbp-3Ch] BYREF

  v11[0] = 0x100000000LL;
  v1 = *(char *)(a1 + 7) < 0;
  v11[1] = 0x800000007LL;
  v12 = 9;
  if ( !v1 )
    return 0;
  v2 = sub_BD2BC0(a1);
  v4 = v2 + v3;
  if ( *(char *)(a1 + 7) < 0 )
    v4 -= sub_BD2BC0(a1);
  v5 = v4 >> 4;
  if ( !(_DWORD)v5 )
    return 0;
  v9 = 16LL * (unsigned int)v5;
  v6 = 0;
  while ( 1 )
  {
    v7 = 0;
    if ( *(char *)(a1 + 7) < 0 )
      v7 = sub_BD2BC0(a1);
    v10 = *(_DWORD *)(*(_QWORD *)(v7 + v6) + 8LL);
    if ( v13 == (_BYTE *)sub_B482E0(v11, (__int64)v13, &v10) )
      break;
    v6 += 16;
    if ( v6 == v9 )
      return 0;
  }
  return (unsigned int)sub_B49240(a1) != 11;
}
