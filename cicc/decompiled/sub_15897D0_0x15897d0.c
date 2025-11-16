// Function: sub_15897D0
// Address: 0x15897d0
//
__int64 __fastcall sub_15897D0(__int64 a1, unsigned int a2, char a3)
{
  __int64 result; // rax
  unsigned int v4; // eax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a3 )
  {
    if ( a2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
LABEL_6:
      *(_DWORD *)(a1 + 24) = a2;
LABEL_7:
      result = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)a1;
      return result;
    }
    sub_16A4EF0(a1, -1, 1);
    v4 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( a2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0;
      goto LABEL_6;
    }
    sub_16A4EF0(a1, 0, 0);
    v4 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 24) = v4;
  if ( v4 <= 0x40 )
    goto LABEL_7;
  return sub_16A4FD0(a1 + 16, a1);
}
