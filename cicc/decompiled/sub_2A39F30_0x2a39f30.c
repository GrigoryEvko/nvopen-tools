// Function: sub_2A39F30
// Address: 0x2a39f30
//
__int64 __fastcall sub_2A39F30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 v6; // rax
  unsigned int v8; // edx
  __int64 v10; // r15
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rax

  if ( *(_DWORD *)(a1 + 8) != 1 )
    return 0;
  v6 = *(unsigned int *)(a2 + 8);
  if ( v6 != 1 )
  {
    if ( a5 < v6 || !*(_DWORD *)(a2 + 8) )
      return 0;
    v8 = *(_DWORD *)(a2 + 8);
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      ++v11;
      v12 = v8;
      if ( v8 <= v11 )
        goto LABEL_13;
      while ( v11 != v10 )
      {
        while ( 1 )
        {
          if ( (unsigned __int8)sub_D0EBA0(
                                  *(_QWORD *)(*(_QWORD *)a2 + 8 * v10),
                                  *(_QWORD *)(*(_QWORD *)a2 + 8 * v11),
                                  0,
                                  a3,
                                  a4) )
            return 0;
          v8 = *(_DWORD *)(a2 + 8);
          ++v11;
          v12 = v8;
          if ( v8 > v11 )
            break;
LABEL_13:
          if ( v12 <= ++v10 || !v8 )
            return 1;
          v11 = 0;
        }
      }
    }
  }
  return 1;
}
