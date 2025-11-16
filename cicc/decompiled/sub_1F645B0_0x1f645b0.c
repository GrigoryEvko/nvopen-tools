// Function: sub_1F645B0
// Address: 0x1f645b0
//
__int64 __fastcall sub_1F645B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r9d

  result = *(unsigned int *)(a2 + 488);
  if ( !(_DWORD)result )
  {
    v3 = *(_QWORD *)(a1 + 80);
    if ( v3 != a1 + 72 )
    {
      v4 = 0x40018000000001LL;
      do
      {
        v5 = 0;
        if ( v3 )
          v5 = v3 - 24;
        v6 = v5;
        v7 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v5) + 16) - 34;
        if ( (unsigned int)v7 <= 0x36 && _bittest64(&v4, v7) )
        {
          v8 = sub_157ED20(v6);
          if ( sub_1F602C0(v8) )
            sub_1F63C50(a2, v8, 0xFFFFFFFF, v9, v10, v11);
        }
        v3 = *(_QWORD *)(v3 + 8);
      }
      while ( a1 + 72 != v3 );
    }
    return sub_1F61D20(a1, a2);
  }
  return result;
}
