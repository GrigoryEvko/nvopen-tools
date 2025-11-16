// Function: sub_2506DA0
// Address: 0x2506da0
//
__int64 __fastcall sub_2506DA0(__int64 *a1, int *a2, __int64 a3)
{
  int v4; // esi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a2;
  v11[0] = a3;
  if ( !(unsigned __int8)sub_A73170(v11, v4) )
    return 0;
  v6 = *a1;
  v7 = sub_A734C0(v11, *a2);
  v10 = *(unsigned int *)(v6 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
  {
    sub_C8D5F0(v6, (const void *)(v6 + 16), v10 + 1, 8u, v8, v9);
    v10 = *(unsigned int *)(v6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v6 + 8 * v10) = v7;
  ++*(_DWORD *)(v6 + 8);
  return 0;
}
