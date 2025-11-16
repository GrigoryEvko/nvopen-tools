// Function: sub_AF4640
// Address: 0xaf4640
//
__int64 __fastcall sub_AF4640(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdx
  __int64 v4; // rax

  if ( !sub_AF4590(a2) )
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v3 = *(_QWORD **)(a2 + 16);
  v4 = (__int64)(*(_QWORD *)(a2 + 24) - (_QWORD)v3) >> 3;
  if ( (_DWORD)v4 )
  {
    if ( *v3 == 4101 )
    {
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v3 + 2;
      *(_QWORD *)(a1 + 8) = v4 - 2;
      return a1;
    }
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v3;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 1;
    return a1;
  }
}
