// Function: sub_E3F500
// Address: 0xe3f500
//
__int64 __fastcall sub_E3F500(__int64 a1, char *a2)
{
  __int64 v3; // r15
  size_t v4; // rax
  void *v5; // rdi
  size_t v6; // r14

  v3 = *(_QWORD *)(a1 + 8);
  if ( !a2 )
    return a1;
  v4 = strlen(a2);
  v5 = *(void **)(v3 + 32);
  v6 = v4;
  if ( v4 <= *(_QWORD *)(v3 + 24) - (_QWORD)v5 )
  {
    if ( v4 )
    {
      memcpy(v5, a2, v4);
      *(_QWORD *)(v3 + 32) += v6;
    }
    return a1;
  }
  sub_CB6200(v3, (unsigned __int8 *)a2, v4);
  return a1;
}
