// Function: sub_22A6D40
// Address: 0x22a6d40
//
__int64 __fastcall sub_22A6D40(__int64 *a1, __int64 a2)
{
  char v2; // bl
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // r13
  char v8; // bl
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ebx
  __int64 v12; // rax
  unsigned __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  __int64 v15; // [rsp+8h] [rbp-28h]

  v5 = *a1;
  v6 = *(__int64 **)(*a1 + 16);
  if ( *(_BYTE *)(*v6 + 8) == 7 || sub_BCAC40(*v6, 8) )
  {
    v2 = sub_AE5020(a2, 0);
    v3 = sub_9208B0(a2, 0);
    v15 = v4;
    v14 = ((1LL << v2) + ((unsigned __int64)(v3 + 7) >> 3) - 1) >> v2 << v2;
    sub_CA1930(&v14);
    BUG();
  }
  v7 = **(_QWORD **)(v5 + 16);
  v8 = sub_AE5020(a2, v7);
  v9 = sub_9208B0(a2, v7);
  v15 = v10;
  v14 = ((1LL << v8) + ((unsigned __int64)(v9 + 7) >> 3) - 1) >> v8 << v8;
  v11 = sub_CA1930(&v14);
  v12 = 0;
  if ( *(_BYTE *)(v7 + 8) == 15 )
    v12 = *(unsigned __int8 *)(sub_AE4AC0(a2, v7) + 16);
  return (v12 << 32) | v11;
}
