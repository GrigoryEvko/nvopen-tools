// Function: sub_3531AE0
// Address: 0x3531ae0
//
__int64 __fastcall sub_3531AE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  bool v3; // dl

  *(_BYTE *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = &unk_4A38F28;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  result = sub_BA91D0(*(_QWORD *)(a2 + 2488), "ptrauth-sign-personality", 0x18u);
  v3 = 0;
  if ( result )
  {
    result = *(_QWORD *)(result + 136);
    v3 = 0;
    if ( result )
    {
      if ( *(_DWORD *)(result + 32) <= 0x40u )
        result = *(_QWORD *)(result + 24);
      else
        result = **(_QWORD **)(result + 24);
      v3 = result == 1;
    }
  }
  *(_BYTE *)(a1 + 72) = v3;
  return result;
}
