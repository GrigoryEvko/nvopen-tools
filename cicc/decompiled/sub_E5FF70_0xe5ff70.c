// Function: sub_E5FF70
// Address: 0xe5ff70
//
unsigned __int64 __fastcall sub_E5FF70(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // r15
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // r14
  _DWORD *v5; // rax
  _DWORD *v6; // rdx
  unsigned int *v8; // rax
  unsigned int *v9; // r12
  unsigned int v10; // esi
  unsigned int *v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx

  v2 = sub_E5FF00(a1, a2);
  v4 = v3;
  v5 = sub_E5F790(a1, a2);
  if ( v5 )
  {
    v6 = v5;
    if ( v5[10] )
    {
      v8 = (unsigned int *)*((_QWORD *)v5 + 4);
      v9 = &v8[4 * v6[12]];
      if ( v8 != v9 )
      {
        while ( 1 )
        {
          v10 = *v8;
          v11 = v8;
          if ( *v8 <= 0xFFFFFFFD )
            break;
          v8 += 4;
          if ( v9 == v8 )
            return v2;
        }
        if ( v9 != v8 )
        {
          while ( 1 )
          {
            v12 = sub_E5FF00(a1, v10);
            if ( v2 > v12 )
              v2 = v12;
            if ( v4 < v13 )
              v4 = v13;
            v11 += 4;
            if ( v11 == v9 )
              break;
            while ( *v11 > 0xFFFFFFFD )
            {
              v11 += 4;
              if ( v9 == v11 )
                return v2;
            }
            if ( v9 == v11 )
              return v2;
            v10 = *v11;
          }
        }
      }
    }
  }
  return v2;
}
