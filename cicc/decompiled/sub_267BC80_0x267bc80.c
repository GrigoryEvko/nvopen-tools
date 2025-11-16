// Function: sub_267BC80
// Address: 0x267bc80
//
__int64 __fastcall sub_267BC80(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  bool v15; // zf
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v3 != 85 || a2 != v3 - 32 )
    return 0;
  v5 = *a1;
  if ( *(char *)(v3 + 7) < 0 )
  {
    v8 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v10 = v8 + v9;
    v11 = 0;
    if ( *(char *)(v3 + 7) < 0 )
      v11 = sub_BD2BC0(v3);
    if ( (unsigned int)((v10 - v11) >> 4) )
      return 0;
  }
  if ( v5 )
  {
    v12 = *(_QWORD *)(v5 + 120);
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(v3 - 32);
    if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *(_QWORD *)(v3 + 80) || v12 != v13 )
      return 0;
  }
  if ( *(_QWORD *)a1[1] == v3 || a3 != a1[2] )
    return 0;
  v14 = a1[3];
  v15 = *(_QWORD *)(v3 + 48) == 0;
  v16[0] = *a1;
  if ( v15 )
    sub_267B7C0(v14, a3, "OMP170", 6u, (__int64)v16);
  else
    sub_267A7C0(v14, v3, "OMP170", 6u, (__int64)v16);
  sub_BD84D0(v3, *(_QWORD *)a1[1]);
  sub_B43D60((_QWORD *)v3);
  *(_BYTE *)a1[4] = 1;
  return 1;
}
