// Function: sub_AADB10
// Address: 0xaadb10
//
__int64 __fastcall sub_AADB10(__int64 a1, unsigned int a2, char a3)
{
  unsigned __int64 v3; // rax
  __int64 result; // rax
  unsigned int v5; // eax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a3 )
  {
    if ( a2 <= 0x40 )
    {
      v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
      if ( !a2 )
        v3 = 0;
      *(_QWORD *)a1 = v3;
      goto LABEL_6;
    }
    sub_C43690(a1, -1, 1);
    v5 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( a2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0;
LABEL_6:
      *(_DWORD *)(a1 + 24) = a2;
LABEL_7:
      result = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)a1;
      return result;
    }
    sub_C43690(a1, 0, 0);
    v5 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 24) = v5;
  if ( v5 <= 0x40 )
    goto LABEL_7;
  return sub_C43780(a1 + 16, a1);
}
