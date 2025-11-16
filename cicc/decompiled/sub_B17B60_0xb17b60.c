// Function: sub_B17B60
// Address: 0xb17b60
//
__int64 __fastcall sub_B17B60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rsi
  _QWORD v8[14]; // [rsp+0h] [rbp-70h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v8[5] = 0x100000000LL;
  v8[6] = a1;
  v8[0] = &unk_49DD210;
  memset(&v8[1], 0, 32);
  sub_CB5980(v8, 0, 0, 0);
  v2 = *(int *)(a2 + 420);
  v3 = *(_QWORD *)(a2 + 80);
  if ( (_DWORD)v2 == -1 )
    v2 = *(unsigned int *)(a2 + 88);
  v4 = v3 + 80 * v2;
  while ( v4 != v3 )
  {
    v5 = *(_QWORD *)(v3 + 40);
    v6 = *(_QWORD *)(v3 + 32);
    v3 += 80;
    sub_CB6200(v8, v6, v5);
  }
  v8[0] = &unk_49DD210;
  sub_CB5840(v8);
  return a1;
}
