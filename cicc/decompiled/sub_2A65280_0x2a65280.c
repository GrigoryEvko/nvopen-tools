// Function: sub_2A65280
// Address: 0x2a65280
//
bool __fastcall sub_2A65280(__int64 *a1, __int64 a2)
{
  unsigned int v2; // edi
  __int64 v3; // rax
  bool result; // al
  __int64 v5; // rax
  __int64 v6; // r12

  if ( *(_BYTE *)a2 > 0x15u )
  {
    v5 = sub_2A64F10(*a1, a2);
    v6 = v5 + 8;
    result = (*(_BYTE *)v5 == 4 || *(_BYTE *)v5 == 5 && (v6 = v5 + 8, sub_9876C0((__int64 *)(v5 + 8))))
          && sub_AB0760(v6);
  }
  else if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    v3 = *(_QWORD *)(a2 + 24);
    if ( v2 > 0x40 )
      v3 = *(_QWORD *)(v3 + 8LL * ((v2 - 1) >> 6));
    return (v3 & (1LL << ((unsigned __int8)v2 - 1))) == 0;
  }
  else
  {
    return 0;
  }
  return result;
}
