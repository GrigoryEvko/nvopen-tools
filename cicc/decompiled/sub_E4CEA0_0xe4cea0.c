// Function: sub_E4CEA0
// Address: 0xe4cea0
//
__int64 __fastcall sub_E4CEA0(_QWORD *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // r13
  unsigned __int8 *v4; // rsi
  size_t v5; // rdx
  void *v6; // rdi
  __int64 result; // rax

  v2 = a1[43];
  if ( v2 )
  {
    v3 = a1[38];
    v4 = (unsigned __int8 *)a1[42];
    v5 = a1[43];
    v6 = *(void **)(v3 + 32);
    if ( v2 > *(_QWORD *)(v3 + 24) - (_QWORD)v6 )
    {
      result = sub_CB6200(v3, v4, v5);
    }
    else
    {
      result = (__int64)memcpy(v6, v4, v5);
      *(_QWORD *)(v3 + 32) += v2;
    }
  }
  a1[43] = 0;
  return result;
}
