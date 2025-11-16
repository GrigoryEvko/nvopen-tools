// Function: sub_2D053E0
// Address: 0x2d053e0
//
__int64 __fastcall sub_2D053E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 result; // rax
  unsigned int v9; // edx

  v6 = (__int64)a3;
  if ( *(_BYTE *)(a1 + 180) )
  {
    v7 = *(__int64 **)(a1 + 160);
    a4 = *(unsigned int *)(a1 + 172);
    a3 = &v7[a4];
    if ( v7 != a3 )
    {
      while ( a2 != *v7 )
      {
        if ( a3 == ++v7 )
          goto LABEL_8;
      }
      return 0;
    }
LABEL_8:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 168) )
    {
      *(_DWORD *)(a1 + 172) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 152);
      goto LABEL_10;
    }
  }
  sub_C8CC70(a1 + 152, a2, (__int64)a3, a4, a5, a6);
  result = v9;
  if ( (_BYTE)v9 )
  {
LABEL_10:
    sub_AE6EC0(*(_QWORD *)(a1 + 48) + 176LL, a2);
    sub_AE6EC0(v6, a2);
    return 1;
  }
  return result;
}
