// Function: sub_15CBBB0
// Address: 0x15cbbb0
//
__int64 __fastcall sub_15CBBB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  size_t v4; // rdx
  const char *v5; // rsi
  void *v6; // rdi
  size_t v8; // [rsp+8h] [rbp-18h]

  v3 = *(_QWORD *)(a1 + 8);
  v5 = (const char *)sub_1649960(a2);
  v6 = *(void **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v6 < v4 )
  {
    sub_16E7EE0(v3, v5);
    return a1;
  }
  else
  {
    if ( v4 )
    {
      v8 = v4;
      memcpy(v6, v5, v4);
      *(_QWORD *)(v3 + 24) += v8;
    }
    return a1;
  }
}
