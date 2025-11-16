// Function: sub_35ABD70
// Address: 0x35abd70
//
void __fastcall sub_35ABD70(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v5 = *(_QWORD **)(a2 + 112);
  v11[0] = (__int64)&unk_50208AC;
  if ( v4 == sub_35ABCB0(v5, (__int64)v4, v11) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_50208AC;
    v7 = *(_QWORD **)(a2 + 112);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v10;
    v4 = &v7[v10];
  }
  v11[0] = (__int64)&unk_501FE44;
  if ( v4 == sub_35ABCB0(v7, (__int64)v4, v11) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_501FE44;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_50209AC);
  sub_2E84680(a1, a2);
}
