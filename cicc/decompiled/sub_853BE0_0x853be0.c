// Function: sub_853BE0
// Address: 0x853be0
//
__int64 __fastcall sub_853BE0(__int64 a1)
{
  int v2; // edi
  __int64 v3; // rax
  __int64 v4; // r12
  int v6[3]; // [rsp+Ch] [rbp-14h] BYREF

  v2 = *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 400);
  if ( v2 == -1 )
  {
    v2 = 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(v3 + 12) == 4 && (*(_BYTE *)(v3 + 17) & 4) != 0 )
      v2 = 0;
  }
  sub_7296F0(v2, v6);
  v4 = sub_869D30();
  sub_729730(v6[0]);
  return v4;
}
