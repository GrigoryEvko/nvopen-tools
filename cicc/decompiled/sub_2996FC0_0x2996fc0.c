// Function: sub_2996FC0
// Address: 0x2996fc0
//
unsigned __int64 __fastcall sub_2996FC0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  _QWORD *v4; // rdi
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

  sub_BB9660(a2, (__int64)&unk_4F89C28);
  sub_BB9660(a2, (__int64)&unk_4F86530);
  sub_BB9660(a2, (__int64)&unk_4F8FAE4);
  v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v4 = *(_QWORD **)(a2 + 112);
  v14[0] = (__int64)&unk_4F86B74;
  if ( v3 == sub_2996F00(v4, (__int64)v3, v14) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F86B74;
    v6 = *(_QWORD **)(a2 + 112);
    v12 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v12;
    v3 = &v6[v12];
  }
  v14[0] = (__int64)&unk_4F8144C;
  if ( v3 == sub_2996F00(v6, (__int64)v3, v14) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, (__int64)v8);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8144C;
    v8 = *(_QWORD **)(a2 + 112);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v13;
    v3 = &v8[v13];
  }
  v14[0] = (__int64)&unk_4F8FBD4;
  result = (unsigned __int64)sub_2996F00(v8, (__int64)v3, v14);
  if ( v3 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v10 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      result = *(_QWORD *)(a2 + 112);
      v3 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8FBD4;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
