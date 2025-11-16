// Function: sub_271E200
// Address: 0x271e200
//
unsigned __int64 __fastcall sub_271E200(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  unsigned __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  __int64 v9; // r8
  _QWORD *v10; // r9
  __int64 v11; // r8
  __int64 v12; // r8
  __int64 v13; // r8
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8FBD4);
  if ( (_BYTE)qword_4FF9C88 )
  {
    v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    v8 = *(_QWORD **)(a2 + 112);
    v14[0] = (__int64)&unk_4F8144C;
    if ( v4 == sub_271E140(v8, (__int64)v4, v14) )
    {
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
      {
        sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, (__int64)v10);
        v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
      }
      *v4 = &unk_4F8144C;
      v10 = *(_QWORD **)(a2 + 112);
      v13 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
      *(_DWORD *)(a2 + 120) = v13;
      v4 = &v10[v13];
    }
    v14[0] = (__int64)&unk_4F8FBD4;
    if ( v4 == sub_271E140(v10, (__int64)v4, v14) )
    {
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
      {
        sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, (__int64)v3);
        v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
      }
      *v4 = &unk_4F8FBD4;
      v3 = *(_QWORD **)(a2 + 112);
      v12 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
      *(_DWORD *)(a2 + 120) = v12;
      v4 = &v3[v12];
    }
  }
  else
  {
    sub_BB9630(a2, (__int64)&unk_4F8FBD4);
    v3 = *(_QWORD **)(a2 + 112);
    v4 = &v3[*(unsigned int *)(a2 + 120)];
  }
  v14[0] = (__int64)&unk_4F86B74;
  result = (unsigned __int64)sub_271E140(v3, (__int64)v4, v14);
  if ( v4 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v6 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      result = *(_QWORD *)(a2 + 112);
      v4 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F86B74;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
