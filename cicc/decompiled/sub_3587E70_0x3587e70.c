// Function: sub_3587E70
// Address: 0x3587e70
//
__int64 __fastcall sub_3587E70(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 (__fastcall **v4)(); // rax

  if ( byte_4F838D4[0] )
  {
    sub_3587420(a1, a2, a3);
    return a1;
  }
  else if ( LOBYTE(qword_503F1C8[8]) && (*(_BYTE *)(*(_QWORD *)(a3 + 16) + 24LL) & 0x10) != 0 )
  {
    v4 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v4;
    *(_DWORD *)a1 = 0;
    return a1;
  }
  else
  {
    sub_3587B20(a1, a2, a3);
    return a1;
  }
}
