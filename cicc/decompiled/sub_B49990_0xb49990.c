// Function: sub_B49990
// Address: 0xb49990
//
bool __fastcall sub_B49990(__int64 a1)
{
  bool v1; // sf
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rax
  int v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+4h] [rbp-3Ch] BYREF
  int v12; // [rsp+Ch] [rbp-34h]
  _BYTE v13[48]; // [rsp+10h] [rbp-30h] BYREF

  v1 = *(char *)(a1 + 7) < 0;
  v11 = 0x800000007LL;
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
  v6 = 0;
  v7 = 16LL * (unsigned int)v5;
  while ( 1 )
  {
    v8 = 0;
    if ( *(char *)(a1 + 7) < 0 )
      v8 = sub_BD2BC0(a1);
    v10 = *(_DWORD *)(*(_QWORD *)(v8 + v6) + 8LL);
    if ( v13 == (_BYTE *)sub_B482E0(&v11, (__int64)v13, &v10) )
      break;
    v6 += 16;
    if ( v6 == v7 )
      return 0;
  }
  return (unsigned int)sub_B49240(a1) != 11;
}
