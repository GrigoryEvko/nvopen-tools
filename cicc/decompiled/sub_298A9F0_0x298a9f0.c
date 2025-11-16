// Function: sub_298A9F0
// Address: 0x298a9f0
//
__int64 __fastcall sub_298A9F0(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rax
  unsigned int v3; // r8d
  __int64 *v5; // rdx
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 *v8; // rbx
  unsigned __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v9[0] = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = sub_298A890(a1 + 624, (__int64 *)v9);
  v3 = 1;
  if ( *(_QWORD *)(a1 + 912) )
  {
    if ( *((_DWORD *)v2 + 4) )
    {
      v5 = (__int64 *)v2[1];
      v6 = &v5[4 * *((unsigned int *)v2 + 6)];
      if ( v5 != v6 )
      {
        while ( 1 )
        {
          v7 = *v5;
          v8 = v5;
          if ( *v5 != -4096 && *v5 != -8192 )
            break;
          v5 += 4;
          if ( v6 == v5 )
            return 0;
        }
        if ( v6 != v5 )
        {
          v3 = 0;
          while ( *(_QWORD *)(a1 + 8) == v8[1] )
          {
            if ( !(_BYTE)v3 )
              v3 = sub_B19720(*(_QWORD *)(a1 + 56), v7, **(_QWORD **)(a1 + 912) & 0xFFFFFFFFFFFFFFF8LL);
            v8 += 4;
            if ( v8 != v6 )
            {
              while ( 1 )
              {
                v7 = *v8;
                if ( *v8 != -4096 && v7 != -8192 )
                  break;
                v8 += 4;
                if ( v6 == v8 )
                  return v3;
              }
              if ( v8 != v6 )
                continue;
            }
            return v3;
          }
        }
      }
    }
    return 0;
  }
  return v3;
}
