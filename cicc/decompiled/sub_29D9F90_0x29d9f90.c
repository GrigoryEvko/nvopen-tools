// Function: sub_29D9F90
// Address: 0x29d9f90
//
__int64 __fastcall sub_29D9F90(__int64 *a1, __int64 a2, __int64 a3)
{
  char v4; // dl
  __int64 result; // rax
  __int64 v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const void *v9; // r13
  const void *v10; // rax
  size_t v11; // rdx
  size_t v12; // r12
  int v13; // eax

  v4 = *(_BYTE *)a3;
  if ( *(_BYTE *)a2 )
  {
    if ( v4 )
    {
      if ( *(_BYTE *)a2 == 1 )
      {
        result = 1;
        if ( v4 == 1 )
        {
          if ( a2 == a3 )
            return 0;
          else
            return sub_29D9730(a1, *(unsigned __int8 **)(a2 + 136), *(unsigned __int8 **)(a3 + 136));
        }
      }
      else
      {
        return (unsigned int)-(v4 == 1);
      }
    }
    else
    {
      return 0xFFFFFFFFLL;
    }
  }
  else
  {
    result = 1;
    if ( !v4 )
    {
      result = 0;
      if ( a2 != a3 )
      {
        v6 = sub_B91420(a2);
        v8 = v7;
        v9 = (const void *)v6;
        v10 = (const void *)sub_B91420(a3);
        v12 = v11;
        if ( v8 <= v11 )
          v11 = v8;
        if ( v11 && (v13 = memcmp(v9, v10, v11)) != 0 )
        {
          return (v13 >> 31) | 1u;
        }
        else
        {
          result = 0;
          if ( v8 != v12 )
            return v8 < v12 ? -1 : 1;
        }
      }
    }
  }
  return result;
}
