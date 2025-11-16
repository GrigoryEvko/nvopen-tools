// Function: sub_1687510
// Address: 0x1687510
//
int __fastcall sub_1687510(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rbx
  unsigned __int64 v3; // r13
  __int64 v4; // rax

  v1 = *(_DWORD *)(a1 + 40);
  if ( v1 >= 0 )
  {
    v2 = 8LL * v1;
    v3 = 8 * (v1 - (unsigned __int64)(unsigned int)v1);
    do
    {
      sub_16856A0(*(_QWORD **)(*(_QWORD *)(a1 + 104) + v2));
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + v2) = 0;
      v4 = v2;
      v2 -= 8;
    }
    while ( v4 != v3 );
  }
  *(_QWORD *)(a1 + 48) = 0;
  sub_16856A0(*(_QWORD **)(a1 + 104));
  sub_16856A0(*(_QWORD **)(a1 + 88));
  sub_16856A0(*(_QWORD **)(a1 + 96));
  return sub_16856A0((_QWORD *)a1);
}
