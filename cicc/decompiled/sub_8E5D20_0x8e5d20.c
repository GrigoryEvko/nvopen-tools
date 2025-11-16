// Function: sub_8E5D20
// Address: 0x8e5d20
//
unsigned __int8 *__fastcall sub_8E5D20(unsigned __int8 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r9
  int v4; // edx
  unsigned __int8 *result; // rax
  _QWORD *v6; // r10
  __int64 v7; // [rsp+8h] [rbp-8h] BYREF

  v3 = a3;
  *a2 = 1;
  v4 = *a1;
  if ( (unsigned int)(v4 - 48) <= 9 )
  {
    a1 = sub_8E5810(a1, &v7, v3);
    if ( v7 < 0 )
    {
      if ( !*(_DWORD *)(v3 + 24) )
      {
        ++*(_QWORD *)(v3 + 32);
        ++*(_QWORD *)(v3 + 48);
        *(_DWORD *)(v3 + 24) = 1;
      }
    }
    else
    {
      *v6 = v7 + 2;
    }
    LOBYTE(v4) = *a1;
  }
  result = a1 + 1;
  if ( (_BYTE)v4 != 95 )
  {
    result = a1;
    if ( !*(_DWORD *)(v3 + 24) )
    {
      ++*(_QWORD *)(v3 + 32);
      ++*(_QWORD *)(v3 + 48);
      *(_DWORD *)(v3 + 24) = 1;
    }
  }
  return result;
}
