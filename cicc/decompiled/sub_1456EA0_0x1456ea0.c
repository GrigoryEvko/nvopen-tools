// Function: sub_1456EA0
// Address: 0x1456ea0
//
__int64 *__fastcall sub_1456EA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 *result; // rax
  __int64 v4; // rcx
  unsigned int v5; // edi
  __int64 *v6; // rax
  __int64 v7; // r8
  int v8; // eax
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 136);
  result = 0;
  if ( (_DWORD)v2 )
  {
    v4 = *(_QWORD *)(a1 + 120);
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + ((unsigned __int64)v5 << 6));
    v7 = *v6;
    if ( *v6 == a2 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + (v2 << 6)) )
        return v6 + 1;
    }
    else
    {
      v8 = 1;
      while ( v7 != -8 )
      {
        v9 = v8 + 1;
        v5 = (v2 - 1) & (v8 + v5);
        v6 = (__int64 *)(v4 + ((unsigned __int64)v5 << 6));
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_3;
        v8 = v9;
      }
    }
    return 0;
  }
  return result;
}
