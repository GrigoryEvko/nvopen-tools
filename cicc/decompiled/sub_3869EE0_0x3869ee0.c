// Function: sub_3869EE0
// Address: 0x3869ee0
//
__int64 __fastcall sub_3869EE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdi
  unsigned int v5; // edx
  __int64 *v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r14
  int v10; // ecx
  int v11; // r10d

  v2 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a2 - 24);
    v4 = *(_QWORD *)(a1 + 8);
    v5 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v6 = (__int64 *)(v4 + 24LL * v5);
    v7 = *v6;
    if ( v3 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 24 * v2) )
      {
        v8 = v6[1];
        if ( *(_BYTE *)(v8 + 16) == 3 && !sub_15E4F60(v6[1]) )
          __asm { jmp     rax }
      }
    }
    else
    {
      v10 = 1;
      while ( v7 != -8 )
      {
        v11 = v10 + 1;
        v5 = (v2 - 1) & (v10 + v5);
        v6 = (__int64 *)(v4 + 24LL * v5);
        v7 = *v6;
        if ( v3 == *v6 )
          goto LABEL_3;
        v10 = v11;
      }
    }
  }
  return 0;
}
