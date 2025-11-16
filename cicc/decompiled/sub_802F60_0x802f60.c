// Function: sub_802F60
// Address: 0x802f60
//
__int64 __fastcall sub_802F60(__int64 a1, __int64 a2, __m128i *a3)
{
  __int64 result; // rax
  __int64 v5; // rdi
  __m128i *v6; // [rsp-30h] [rbp-30h]

  result = dword_4D04798;
  if ( dword_4D04798 )
  {
    result = *(_BYTE *)(a1 + 48) & 0xFB;
    if ( (*(_BYTE *)(a1 + 48) & 0xFB) == 2 )
    {
      v5 = *(_QWORD *)(a1 + 56);
      if ( *(_BYTE *)(v5 + 173) == 10 )
      {
        v6 = a3;
        sub_7E31E0(v5);
        a3 = v6;
      }
      return sub_802E80(v5, a1, a2, a3);
    }
  }
  return result;
}
