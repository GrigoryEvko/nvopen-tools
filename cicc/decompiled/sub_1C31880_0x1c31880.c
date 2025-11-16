// Function: sub_1C31880
// Address: 0x1c31880
//
__int64 __fastcall sub_1C31880(__int64 a1)
{
  _BYTE *v1; // rax
  __int64 result; // rax
  _QWORD *v3; // r12

  v1 = *(_BYTE **)(a1 + 16);
  if ( v1 )
    *v1 = 0;
  result = *(unsigned int *)(a1 + 4);
  if ( !(_DWORD)result )
  {
    v3 = *(_QWORD **)(a1 + 24);
    if ( v3[3] != v3[1] )
    {
      sub_16E7BA0(*(__int64 **)(a1 + 24));
      v3 = *(_QWORD **)(a1 + 24);
      if ( v3[3] != v3[1] )
        sub_16E7BA0(*(__int64 **)(a1 + 24));
    }
    return sub_1C3EF50(v3[5]);
  }
  return result;
}
