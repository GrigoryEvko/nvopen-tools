// Function: sub_14D1290
// Address: 0x14d1290
//
__int64 __fastcall sub_14D1290(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v9; // rax

  v2 = *a1;
  v3 = sub_1649C60(a1);
  v4 = *(_QWORD *)v3;
  v5 = *(__int64 **)(*(_QWORD *)v3 + 16LL);
  v6 = *v5;
  *a2 = *v5;
  v7 = *(_DWORD *)(v2 + 8) >> 8;
  if ( *(_DWORD *)(v4 + 8) >> 8 == (_DWORD)v7 )
    return v3;
  v9 = sub_1647190(v6, v7);
  return sub_15A4A70(v3, v9);
}
