// Function: sub_254A530
// Address: 0x254a530
//
__int64 __fastcall sub_254A530(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v6; // r8d

  v2 = 1;
  if ( !*(_BYTE *)(a1 + 97) )
    return v2;
  v2 = *(unsigned __int8 *)(a1 + 132);
  if ( (_BYTE)v2 )
  {
    v4 = *(_QWORD **)(a1 + 112);
    v5 = &v4[*(unsigned int *)(a1 + 124)];
    if ( v4 != v5 )
    {
      while ( a2 != *v4 )
      {
        if ( v5 == ++v4 )
          return 0;
      }
      return v2;
    }
    return 0;
  }
  else
  {
    LOBYTE(v6) = sub_C8CA60(a1 + 104, a2) != 0;
    return v6;
  }
}
