// Function: sub_7A80F0
// Address: 0x7a80f0
//
__int64 ***__fastcall sub_7A80F0(_QWORD *a1)
{
  __int64 ***result; // rax
  __int64 **v2; // rbx
  __int64 **v3; // rdx
  __int64 v4; // rcx

  result = *(__int64 ****)(*a1 + 168LL);
  v2 = *result;
  if ( *result )
  {
    while ( 1 )
    {
      result = (__int64 ***)sub_7A80B0((__int64)v2[5]);
      if ( !(_DWORD)result )
        goto LABEL_8;
      result = (__int64 ***)((char *)v2[13] + v2[5][16]);
      if ( (unsigned __int64)result <= a1[1] )
        goto LABEL_8;
      if ( !dword_4D0425C )
        goto LABEL_7;
      if ( ((_BYTE)v2[12] & 2) == 0 )
        break;
      if ( ((_BYTE)v2[12] & 1) == 0 )
        goto LABEL_8;
      v3 = (__int64 **)v2[14];
      if ( v3 && *v3 )
        goto LABEL_13;
LABEL_7:
      a1[1] = result;
      a1[2] = 0;
LABEL_8:
      v2 = (__int64 **)*v2;
      if ( !v2 )
        return result;
    }
    v3 = (__int64 **)v2[14];
    if ( (*(_BYTE *)(v3[1][2] + 96) & 2) == 0 )
      goto LABEL_7;
    if ( ((_BYTE)v2[12] & 1) == 0 )
      goto LABEL_8;
    if ( !*v3 )
      goto LABEL_7;
LABEL_13:
    v4 = v2[7][20];
    if ( v4 )
    {
      while ( (*(_BYTE *)(v4 + 144) & 4) != 0 && !*(_BYTE *)(v4 + 137) )
      {
        v4 = *(_QWORD *)(v4 + 112);
        if ( !v4 )
          goto LABEL_7;
      }
      while ( ((_BYTE)v3[3] & 1) != 0 || (*(_BYTE *)(v3[1][2] + 96) & 2) == 0 )
      {
        v3 = (__int64 **)*v3;
        if ( !v3 )
          goto LABEL_7;
      }
      goto LABEL_8;
    }
    goto LABEL_7;
  }
  return result;
}
