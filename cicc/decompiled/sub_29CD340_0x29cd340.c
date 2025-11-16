// Function: sub_29CD340
// Address: 0x29cd340
//
unsigned __int64 __fastcall sub_29CD340(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  unsigned __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v11[0] = (__int64)&unk_4F86B74;
  if ( v4 == sub_29CD280(v3, (__int64)v4, v11) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F86B74;
    v6 = *(_QWORD **)(a2 + 112);
    v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v10;
    v4 = &v6[v10];
  }
  v11[0] = (__int64)&unk_4F8144C;
  result = (unsigned __int64)sub_29CD280(v6, (__int64)v4, v11);
  if ( v4 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v8 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      result = *(_QWORD *)(a2 + 112);
      v4 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
