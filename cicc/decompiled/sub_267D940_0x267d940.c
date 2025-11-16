// Function: sub_267D940
// Address: 0x267d940
//
__int64 __fastcall sub_267D940(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 != 85 || a2 != v2 - 32 )
    return 0;
  v4 = *a1;
  if ( *(char *)(v2 + 7) < 0 )
  {
    v5 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v7 = v5 + v6;
    v8 = 0;
    if ( *(char *)(v2 + 7) < 0 )
      v8 = sub_BD2BC0(v2);
    if ( (unsigned int)((v7 - v8) >> 4) )
      return 0;
  }
  if ( v4 )
  {
    v9 = *(_QWORD *)(v4 + 120);
    if ( !v9 )
      return 0;
    v10 = *(_QWORD *)(v2 - 32);
    if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v2 + 80) || v9 != v10 )
      return 0;
  }
  sub_267D3C0(a1[1], v2, "OMP112", 6u);
  return 0;
}
