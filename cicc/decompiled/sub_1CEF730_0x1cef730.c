// Function: sub_1CEF730
// Address: 0x1cef730
//
__int64 __fastcall sub_1CEF730(unsigned int **a1, __int64 a2)
{
  __int64 *v3; // rax
  struct __jmp_buf_tag *v4; // r12
  __int64 *v5; // r13
  _BYTE *v6; // rax
  __int64 i; // r14
  __int64 j; // r15
  __int64 k; // rbx
  __int64 v10; // rcx
  char v11; // al

  if ( (unsigned __int8)sub_1CEE970(a1, a2) )
  {
    v3 = sub_1C3E710();
    v4 = (struct __jmp_buf_tag *)sub_16D40F0((__int64)v3);
    if ( v4 )
    {
      v5 = sub_1C3E7B0();
      v6 = (_BYTE *)sub_1C42D70(1, 1);
      *v6 = 1;
      sub_16D40E0((__int64)v5, v6);
      longjmp(v4, 1);
    }
  }
  if ( byte_4FC0708[0] )
  {
    for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      for ( j = *(_QWORD *)(i + 24); i + 16 != j; j = *(_QWORD *)(j + 8) )
      {
        if ( !j )
          BUG();
        for ( k = *(_QWORD *)(j + 24); j + 16 != k; k = *(_QWORD *)(k + 8) )
        {
          if ( !k )
            BUG();
          v10 = *(_QWORD *)(k - 24);
          v11 = *(_BYTE *)(v10 + 8);
          if ( v11 == 16 )
            v11 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
          if ( (unsigned __int8)(v11 - 1) <= 5u || *(_BYTE *)(k - 8) == 76 )
            sub_15F23E0(k - 24, 1);
        }
      }
    }
  }
  return 0;
}
