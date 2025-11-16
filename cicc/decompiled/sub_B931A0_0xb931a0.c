// Function: sub_B931A0
// Address: 0xb931a0
//
__int64 __fastcall sub_B931A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned __int8 **v10; // rbx
  unsigned __int8 **i; // r12
  unsigned __int8 *v12; // rdi

  result = *(_BYTE *)(a1 + 1) & 0x7F;
  if ( (_BYTE)result == 2 || (a3 = *(unsigned int *)(a1 - 8), (_DWORD)a3) )
  {
    sub_B93190(a1, a2, a3, a4, a5);
    result = *(unsigned __int8 *)(a1 - 16);
    if ( (result & 2) != 0 )
    {
      v10 = *(unsigned __int8 ***)(a1 - 32);
      v9 = *(unsigned int *)(a1 - 24);
    }
    else
    {
      result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
      v9 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
      v10 = (unsigned __int8 **)(a1 - result - 16);
    }
    for ( i = &v10[v9]; i != v10; ++v10 )
    {
      v12 = *v10;
      if ( *v10 )
      {
        result = (unsigned int)*v12 - 5;
        if ( (unsigned __int8)(*v12 - 5) <= 0x1Fu )
        {
          if ( (v12[1] & 0x7F) == 2 || (result = *((unsigned int *)v12 - 2), (_DWORD)result) )
            result = sub_B931A0(v12, a2, v9, v6, v7, v8);
        }
      }
    }
  }
  return result;
}
