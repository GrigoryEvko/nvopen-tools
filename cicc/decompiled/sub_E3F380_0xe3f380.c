// Function: sub_E3F380
// Address: 0xe3f380
//
__int64 __fastcall sub_E3F380(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // r13
  unsigned __int8 *v4; // rax
  size_t v5; // rdx
  void *v6; // rdi
  size_t v8; // [rsp+8h] [rbp-18h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( (a2[7] & 0x10) != 0 )
  {
    v4 = (unsigned __int8 *)sub_BD5D20((__int64)a2);
    v6 = *(void **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v6 < v5 )
    {
      sub_CB6200(v3, v4, v5);
      return a1;
    }
    else
    {
      if ( v5 )
      {
        v8 = v5;
        memcpy(v6, v4, v5);
        *(_QWORD *)(v3 + 32) += v8;
      }
      return a1;
    }
  }
  else
  {
    sub_A5BF40(a2, *(_QWORD *)(a1 + 8), 0, 0);
    return a1;
  }
}
