// Function: sub_18E87D0
// Address: 0x18e87d0
//
__int64 __fastcall sub_18E87D0(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rbx
  __int64 *v3; // r12
  __int64 v4; // rbx

  result = *(_BYTE *)(a1 + 64) & 1;
  if ( *(_DWORD *)(a1 + 64) >> 1 )
  {
    if ( (_BYTE)result )
    {
      v2 = (__int64 *)(a1 + 72);
      v3 = (__int64 *)(a1 + 136);
    }
    else
    {
      v2 = *(__int64 **)(a1 + 72);
      v3 = &v2[2 * *(unsigned int *)(a1 + 80)];
      if ( v2 == v3 )
        return result;
    }
    do
    {
      result = *v2;
      if ( *v2 != -8 && result != -16 )
        break;
      v2 += 2;
    }
    while ( v3 != v2 );
  }
  else
  {
    if ( (_BYTE)result )
    {
      v4 = a1 + 72;
      result = 64;
    }
    else
    {
      v4 = *(_QWORD *)(a1 + 72);
      result = 16LL * *(unsigned int *)(a1 + 80);
    }
    v2 = (__int64 *)(result + v4);
    v3 = v2;
  }
  if ( v2 != v3 )
  {
LABEL_8:
    if ( !*(_QWORD *)(*v2 + 8) )
      result = sub_15F20C0((_QWORD *)*v2);
    while ( 1 )
    {
      v2 += 2;
      if ( v2 == v3 )
        break;
      result = *v2;
      if ( *v2 != -16 && result != -8 )
      {
        if ( v3 != v2 )
          goto LABEL_8;
        return result;
      }
    }
  }
  return result;
}
