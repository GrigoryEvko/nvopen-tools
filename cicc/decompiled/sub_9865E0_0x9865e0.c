// Function: sub_9865E0
// Address: 0x9865e0
//
__int64 __fastcall sub_9865E0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // rax

  v2 = ~(1LL << ((unsigned __int8)a2 - 1));
  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 <= 0x40 )
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a2;
    if ( !a2 )
      v3 = 0;
    *(_QWORD *)a1 = v3;
    goto LABEL_5;
  }
  sub_C43690(a1, -1, 1);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
LABEL_5:
    *(_QWORD *)a1 &= v2;
    return a1;
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * ((a2 - 1) >> 6)) &= v2;
  return a1;
}
