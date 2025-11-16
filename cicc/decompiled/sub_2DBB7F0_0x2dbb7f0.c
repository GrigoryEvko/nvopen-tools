// Function: sub_2DBB7F0
// Address: 0x2dbb7f0
//
unsigned __int64 __fastcall sub_2DBB7F0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  _QWORD *v4; // rdi
  __int64 v5; // r8
  _QWORD *v6; // r9
  unsigned __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_5027190);
  v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v4 = *(_QWORD **)(a2 + 112);
  v11[0] = (__int64)&unk_4F86530;
  if ( v3 == sub_2DBB730(v4, (__int64)v3, v11) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F86530;
    v6 = *(_QWORD **)(a2 + 112);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v10;
    v3 = &v6[v10];
  }
  v11[0] = (__int64)&unk_4F86B74;
  result = (unsigned __int64)sub_2DBB730(v6, (__int64)v3, v11);
  if ( v3 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v8 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      result = *(_QWORD *)(a2 + 112);
      v3 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F86B74;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
