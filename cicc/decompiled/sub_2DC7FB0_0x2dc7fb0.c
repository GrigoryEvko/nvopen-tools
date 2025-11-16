// Function: sub_2DC7FB0
// Address: 0x2dc7fb0
//
__int64 __fastcall sub_2DC7FB0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rdi
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v11; // r8
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  v4 = *(_QWORD **)(a2 + 112);
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  v12[0] = (__int64)&unk_50208AC;
  if ( v5 == sub_2DC7EF0(v4, (__int64)v5, v12) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_50208AC;
    v7 = *(_QWORD **)(a2 + 112);
    v11 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v11;
    v5 = &v7[v11];
  }
  v12[0] = (__int64)&unk_501FE44;
  if ( v5 == sub_2DC7EF0(v7, (__int64)v5, v12) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  return sub_2E84680(a1, a2);
}
