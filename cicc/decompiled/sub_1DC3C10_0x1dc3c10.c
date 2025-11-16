// Function: sub_1DC3C10
// Address: 0x1dc3c10
//
__int64 __fastcall sub_1DC3C10(__int64 a1, __int64 *a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rbx

  result = *(_QWORD *)(a1 + 8);
  if ( a3 < 0 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(result + 24) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  }
  else
  {
    result = *(_QWORD *)(result + 272);
    v4 = *(_QWORD *)(result + 8LL * (unsigned int)a3);
  }
  if ( v4 )
  {
    if ( (*(_BYTE *)(v4 + 3) & 0x10) != 0 )
    {
      do
      {
        result = sub_1DC3350(*(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 32), a2, v4);
        v4 = *(_QWORD *)(v4 + 32);
        if ( !v4 )
          break;
LABEL_6:
        ;
      }
      while ( (*(_BYTE *)(v4 + 3) & 0x10) != 0 );
    }
    else
    {
      v4 = *(_QWORD *)(v4 + 32);
      if ( v4 )
        goto LABEL_6;
    }
  }
  return result;
}
