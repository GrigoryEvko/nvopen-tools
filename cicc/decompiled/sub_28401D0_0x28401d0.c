// Function: sub_28401D0
// Address: 0x28401d0
//
unsigned __int64 __fastcall sub_28401D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  __int64 *v5; // rbx
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  int v9; // edx

  v4 = (__int64 *)(a3 + 8 * a4);
  if ( (__int64 *)a3 == v4 )
  {
LABEL_6:
    v7 = *(_QWORD *)(a1 + 56);
    v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 == v7 + 48 )
    {
      return 0;
    }
    else
    {
      if ( !v8 )
        BUG();
      v9 = *(unsigned __int8 *)(v8 - 24);
      result = v8 - 24;
      if ( (unsigned int)(v9 - 30) >= 0xB )
        return 0;
    }
  }
  else
  {
    v5 = (__int64 *)a3;
    while ( (unsigned __int8)sub_D48480(*(_QWORD *)(a1 + 40), *v5, a3, a4) )
    {
      if ( v4 == ++v5 )
        goto LABEL_6;
    }
    return a2;
  }
  return result;
}
