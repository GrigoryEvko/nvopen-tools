// Function: sub_2F29660
// Address: 0x2f29660
//
unsigned __int64 __fastcall sub_2F29660(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r8
  _QWORD *v5; // rsi
  unsigned __int64 result; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rdi
  __int64 v10; // r8
  _QWORD *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_2E84680(a1, a2);
  sub_BB9660(a2, (__int64)&unk_50208AC);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v14[0] = (__int64)&unk_50208AC;
  v5 = &v3[v4];
  result = (unsigned __int64)sub_2F29360(v3, (__int64)v5, v14);
  if ( v5 != (_QWORD *)result )
  {
    if ( !(_BYTE)qword_5022BE8 )
      return result;
    goto LABEL_7;
  }
  result = *(unsigned int *)(a2 + 124);
  if ( v7 + 1 > result )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, v8);
    result = *(_QWORD *)(a2 + 112);
    v5 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
  }
  *v5 = &unk_50208AC;
  ++*(_DWORD *)(a2 + 120);
  if ( (_BYTE)qword_5022BE8 )
  {
LABEL_7:
    sub_BB9660(a2, (__int64)&unk_501FE44);
    v9 = *(_QWORD **)(a2 + 112);
    v10 = *(unsigned int *)(a2 + 120);
    v14[0] = (__int64)&unk_501FE44;
    v11 = &v9[v10];
    result = (unsigned __int64)sub_2F29360(v9, (__int64)v11, v14);
    if ( v11 == (_QWORD *)result )
    {
      result = *(unsigned int *)(a2 + 124);
      if ( v12 + 1 > result )
      {
        sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, v13);
        result = *(_QWORD *)(a2 + 112);
        v11 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
      }
      *v11 = &unk_501FE44;
      ++*(_DWORD *)(a2 + 120);
    }
  }
  return result;
}
