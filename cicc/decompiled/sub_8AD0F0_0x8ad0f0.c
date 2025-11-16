// Function: sub_8AD0F0
// Address: 0x8ad0f0
//
__int64 __fastcall sub_8AD0F0(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 *v3; // rbx
  __int64 i; // rbx
  int v5; // esi
  __int64 j; // rbx

  v1 = *(_QWORD *)(a1 + 168);
  result = *(_QWORD *)(v1 + 152);
  if ( result && (*(_BYTE *)(result + 29) & 0x20) == 0 )
  {
    v3 = *(__int64 **)(result + 144);
    if ( v3 )
    {
      do
      {
        while ( *((_BYTE *)v3 + 174) == 7 || !*(_QWORD *)(*v3 + 96) || (*((_BYTE *)v3 + 195) & 8) != 0 )
        {
          v3 = (__int64 *)v3[14];
          if ( !v3 )
            goto LABEL_10;
        }
        sub_8AD0D0(*v3, 0, 1);
        v3 = (__int64 *)v3[14];
      }
      while ( v3 );
LABEL_10:
      result = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
    }
    for ( i = *(_QWORD *)(result + 112); i; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_BYTE *)(i + 170) & 0x60) == 0 && *(_QWORD *)(*(_QWORD *)i + 96LL) )
      {
        v5 = dword_4F077BC;
        if ( dword_4F077BC )
          v5 = (*(_BYTE *)(i + 168) & 0x40) != 0;
        sub_8AD0D0(*(_QWORD *)i, v5, 1);
      }
    }
    result = *(_QWORD *)(v1 + 152);
    for ( j = *(_QWORD *)(result + 104); j; j = *(_QWORD *)(j + 112) )
    {
      result = (unsigned int)*(unsigned __int8 *)(j + 140) - 9;
      if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u )
        result = sub_8AD0F0(j);
    }
  }
  return result;
}
