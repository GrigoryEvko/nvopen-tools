// Function: sub_5D2590
// Address: 0x5d2590
//
__int64 __fastcall sub_5D2590(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v6; // rdi
  char v7; // al
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = *a1;
  if ( !*a1 )
    return 0;
  v3 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v1 + 8) == 3 )
      {
        if ( (unsigned int)sub_5D19D0(v1) )
        {
          v4 = 0x7FFFFFFFFFFFFFFFLL;
          v8[0] = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v6 = *(_QWORD *)(v1 + 32);
          v8[0] = 0;
          v7 = *(_BYTE *)(v6 + 10);
          if ( v7 == 3 )
          {
            sub_5CACA0(v6, v1, 0, 0x7FFFFFFFFFFFFFFFLL, v8);
            v4 = v8[0];
          }
          else
          {
            if ( v7 != 4 )
              sub_721090(v6);
            v4 = *(unsigned int *)(*(_QWORD *)(v6 + 40) + 136LL);
            v8[0] = v4;
          }
        }
        if ( v4 > v3 )
          break;
      }
      v1 = *(_QWORD *)v1;
      if ( !v1 )
        return v3;
    }
    *a1 = v1;
    v1 = *(_QWORD *)v1;
    v3 = v4;
  }
  while ( v1 );
  return v3;
}
