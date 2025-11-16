// Function: sub_2E79BC0
// Address: 0x2e79bc0
//
__int64 __fastcall sub_2E79BC0(__int64 a1, __int64 a2)
{
  __int64 **v2; // r8
  __int64 **v3; // r10
  unsigned int v4; // r9d
  __int64 *v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 *v13; // rdx

  v2 = *(__int64 ***)(a1 + 8);
  v3 = *(__int64 ***)(a1 + 16);
  v4 = 0;
  while ( v2 != v3 )
  {
    v5 = v2[1];
    v6 = *v2;
    v7 = (char *)v5 - (char *)*v2;
    v8 = v7 >> 5;
    v9 = v7 >> 3;
    if ( v8 <= 0 )
    {
LABEL_19:
      if ( v9 == 2 )
        goto LABEL_29;
      if ( v9 != 3 )
      {
        if ( v9 != 1 || a2 != *v6 )
          goto LABEL_16;
        goto LABEL_9;
      }
      if ( a2 != *v6 )
      {
        ++v6;
LABEL_29:
        if ( a2 != *v6 && a2 != *++v6 )
          goto LABEL_16;
      }
    }
    else
    {
      v10 = &v6[4 * v8];
      while ( a2 != *v6 )
      {
        if ( a2 == v6[1] )
        {
          ++v6;
          break;
        }
        if ( a2 == v6[2] )
        {
          v6 += 2;
          break;
        }
        if ( a2 == v6[3] )
        {
          v6 += 3;
          break;
        }
        v6 += 4;
        if ( v10 == v6 )
        {
          v9 = v5 - v6;
          goto LABEL_19;
        }
      }
    }
LABEL_9:
    if ( v6 != v5 )
    {
      v11 = v6 + 1;
      if ( v5 == v6 + 1 )
      {
        v4 = 1;
      }
      else
      {
        do
        {
          v12 = *v11;
          if ( a2 != *v11 )
            *v6++ = v12;
          ++v11;
        }
        while ( v5 != v11 );
        v13 = v2[1];
        LOBYTE(v12) = v6 != v13;
        v4 |= v12;
        if ( v6 == v13 )
          goto LABEL_16;
      }
      v2[1] = v6;
    }
LABEL_16:
    v2 += 4;
  }
  return v4;
}
