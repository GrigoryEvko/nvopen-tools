// Function: sub_AA3950
// Address: 0xaa3950
//
__int64 __fastcall sub_AA3950(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int8 *v7; // rdi
  int v8; // eax
  unsigned __int64 v9; // rax
  unsigned __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  result = sub_A910B0(a1, (__int64 *)v10, 1);
  if ( (_BYTE)result )
  {
    v4 = 0x8000000000041LL;
    v5 = *(_QWORD *)(a1 + 16);
    while ( v5 )
    {
      v6 = v5;
      v5 = *(_QWORD *)(v5 + 8);
      v7 = *(unsigned __int8 **)(v6 + 24);
      v8 = *v7;
      if ( (unsigned __int8)v8 > 0x1Cu )
      {
        v9 = (unsigned int)(v8 - 34);
        if ( (unsigned __int8)v9 <= 0x33u )
        {
          if ( _bittest64(&v4, v9) )
            sub_A939D0(v7, v10[0], v3);
        }
      }
    }
    return sub_B2E860(a1);
  }
  return result;
}
