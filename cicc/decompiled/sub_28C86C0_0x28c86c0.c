// Function: sub_28C86C0
// Address: 0x28c86c0
//
__int64 __fastcall sub_28C86C0(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // r8
  __int64 result; // rax
  int v5; // esi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rdx
  int v10; // edx
  int v11; // r10d

  v2 = *(_DWORD *)(a1 + 1456);
  v3 = *(_QWORD *)(a1 + 1440);
  result = a2;
  if ( v2 )
  {
    v5 = v2 - 1;
    v6 = (v2 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( result == *v7 )
    {
LABEL_3:
      v9 = v7[1];
      if ( v9 )
      {
        if ( *(_QWORD *)(a1 + 1392) == v9 )
        {
          return sub_ACADE0(*(__int64 ***)(result + 8));
        }
        else
        {
          result = *(_QWORD *)(v9 + 40);
          if ( !result )
            return *(_QWORD *)(v9 + 8);
        }
      }
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v11 = v10 + 1;
        v6 = v5 & (v10 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( result == *v7 )
          goto LABEL_3;
        v10 = v11;
      }
    }
  }
  return result;
}
