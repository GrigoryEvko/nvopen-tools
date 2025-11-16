// Function: sub_895F30
// Address: 0x895f30
//
__int64 __fastcall sub_895F30(__int64 a1)
{
  char v1; // al
  _BYTE *v2; // rax
  _BYTE *v3; // r12
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 1 )
  {
    v2 = sub_724D80(0);
    *(_QWORD *)(a1 + 32) = v2;
    v3 = v2;
    result = sub_72C930();
    *((_QWORD *)v3 + 16) = result;
    *(_BYTE *)(a1 + 25) |= 2u;
  }
  else if ( v1 == 2 )
  {
    result = *(_QWORD *)(*(_QWORD *)(sub_87F550() + 88) + 104LL);
    *(_BYTE *)(a1 + 25) |= 2u;
    *(_QWORD *)(a1 + 32) = result;
  }
  else
  {
    if ( v1 )
      sub_721090();
    result = sub_72C930();
    *(_BYTE *)(a1 + 25) |= 2u;
    *(_QWORD *)(a1 + 32) = result;
  }
  return result;
}
