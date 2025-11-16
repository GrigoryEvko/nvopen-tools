// Function: sub_F420B0
// Address: 0xf420b0
//
unsigned __int64 __fastcall sub_F420B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // r9
  unsigned __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r8
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v14[0] = (__int64)&unk_4F8144C;
  if ( v4 == sub_F41FF0(v3, (__int64)v4, v14) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    v6 = *(_QWORD **)(a2 + 112);
    v12 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v12;
    v4 = &v6[v12];
  }
  v14[0] = (__int64)&unk_4F875EC;
  if ( v4 == sub_F41FF0(v6, (__int64)v4, v14) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, (__int64)v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F875EC;
    v8 = *(_QWORD **)(a2 + 112);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v13;
    v4 = &v8[v13];
  }
  v14[0] = (__int64)&unk_4F8C12C;
  result = (unsigned __int64)sub_F41FF0(v8, (__int64)v4, v14);
  if ( v4 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v10 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      result = *(_QWORD *)(a2 + 112);
      v4 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8C12C;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
