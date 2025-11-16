// Function: sub_34BC140
// Address: 0x34bc140
//
void __fastcall sub_34BC140(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  *(_BYTE *)(a2 + 160) = 1;
  sub_BB9660(a2, (__int64)&unk_501695C);
  v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
  v5 = *(_QWORD **)(a2 + 144);
  v11[0] = (__int64)&unk_501FE44;
  if ( v4 == sub_34BC080(v5, (__int64)v4, v11) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v6 + 1, 8u, v6, (__int64)v7);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_501FE44;
    v7 = *(_QWORD **)(a2 + 144);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v10;
    v4 = &v7[v10];
  }
  v11[0] = (__int64)&unk_50209DC;
  if ( v4 == sub_34BC080(v7, (__int64)v4, v11) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v8 + 1, 8u, v8, v9);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v4 = &unk_50209DC;
    ++*(_DWORD *)(a2 + 152);
  }
  sub_2E84680(a1, a2);
}
