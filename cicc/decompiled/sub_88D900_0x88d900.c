// Function: sub_88D900
// Address: 0x88d900
//
__int64 __fastcall sub_88D900(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rdx
  __int64 *v4[2]; // [rsp+8h] [rbp-18h] BYREF

  v4[0] = (__int64 *)a1;
  if ( !a1 )
    return 1;
  v1 = (__int64 *)a1;
  if ( *(_BYTE *)(a1 + 8) == 3 )
  {
    sub_72F220(v4);
    v1 = v4[0];
    if ( !v4[0] )
      return 1;
  }
  while ( 1 )
  {
    v2 = v1[2];
    v1 = (__int64 *)*v1;
    v4[0] = v1;
    if ( v2 )
      break;
    if ( v1 )
    {
      if ( *((_BYTE *)v1 + 8) != 3 )
        continue;
      sub_72F220(v4);
      v1 = v4[0];
      if ( v4[0] )
        continue;
    }
    return 1;
  }
  if ( v1 && (*((_BYTE *)v1 + 8) != 3 || (sub_72F220(v4), v4[0])) )
    return 0;
  else
    return 1;
}
