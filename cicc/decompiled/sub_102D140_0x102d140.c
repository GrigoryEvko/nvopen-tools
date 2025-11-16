// Function: sub_102D140
// Address: 0x102d140
//
void __fastcall sub_102D140(__int64 a1, __int64 a2)
{
  const char *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( (_BYTE)qword_4F8F0C8 )
  {
    v2 = sub_BD5D20(a2);
    v6[1] = v3;
    v6[0] = (__int64)v2;
    if ( sub_C931B0(v6, "cutlass", 7u, 0) == -1 )
      sub_102A040(a1, a2, v4, v5);
  }
}
