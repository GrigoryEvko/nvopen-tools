// Function: sub_3004D00
// Address: 0x3004d00
//
void __fastcall sub_3004D00(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v10[0] = (__int64)&unk_50208AC;
  if ( v4 == sub_3004BB0(v3, (__int64)v4, v10) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_50208AC;
    v6 = *(_QWORD **)(a2 + 112);
    v9 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v9;
    v4 = &v6[v9];
  }
  v10[0] = (__int64)&unk_501FE44;
  if ( v4 == sub_3004BB0(v6, (__int64)v4, v10) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_2E84680(a1, a2);
}
