// Function: sub_1D03530
// Address: 0x1d03530
//
__int64 __fastcall sub_1D03530(_QWORD *a1)
{
  __int64 *v1; // r14
  __int64 *v2; // r12
  __int64 *v3; // rbx
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // al
  __int64 result; // rax

  v1 = (__int64 *)a1[3];
  v2 = (__int64 *)a1[2];
  if ( v1 == v2 )
    return 0;
  v3 = v2 + 1;
  if ( v1 != v2 + 1 )
  {
    do
    {
      while ( 1 )
      {
        v4 = (*(_BYTE *)(*v2 + 229) & 0x10) != 0;
        v5 = (*(_BYTE *)(*v3 + 229) & 0x10) != 0;
        if ( v4 == v5 )
          break;
        if ( v4 < v5 )
          v2 = v3;
        if ( v1 == ++v3 )
          goto LABEL_11;
      }
      if ( sub_1D03130(*v2, *v3, a1[21]) )
        v2 = v3;
      ++v3;
    }
    while ( v1 != v3 );
LABEL_11:
    v3 = (__int64 *)a1[3];
  }
  result = *v2;
  if ( v2 != v3 - 1 )
  {
    *v2 = *(v3 - 1);
    *(v3 - 1) = result;
    v2 = (__int64 *)(a1[3] - 8LL);
  }
  a1[3] = v2;
  *(_DWORD *)(result + 196) = 0;
  return result;
}
