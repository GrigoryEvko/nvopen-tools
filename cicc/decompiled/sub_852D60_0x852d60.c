// Function: sub_852D60
// Address: 0x852d60
//
_DWORD *sub_852D60()
{
  _DWORD *result; // rax
  __int64 v1; // rbx
  int v2; // r12d
  int v3; // edi

  result = dword_4D03C98;
  dword_4D03C98[0] = 0;
  if ( dword_4F073A8 > 1 )
  {
    v1 = 16;
    v2 = 2;
    do
    {
      while ( 1 )
      {
        result = qword_4F073B0;
        if ( *(_QWORD *)((char *)qword_4F073B0 + v1) )
        {
          result = *(_DWORD **)((char *)qword_4F072B0 + v1);
          if ( result[60] == -1 )
            break;
        }
        ++v2;
        v1 += 8;
        if ( dword_4F073A8 < v2 )
          return result;
      }
      v3 = v2;
      v1 += 8;
      ++v2;
      result = (_DWORD *)sub_823780(v3);
    }
    while ( dword_4F073A8 >= v2 );
  }
  return result;
}
