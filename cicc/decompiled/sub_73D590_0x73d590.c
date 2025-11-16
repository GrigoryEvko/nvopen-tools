// Function: sub_73D590
// Address: 0x73d590
//
__int64 __fastcall sub_73D590(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  char v4; // r12
  __m128i *v5; // rax

  result = (__int64)&dword_4F0690C;
  if ( dword_4F0690C )
  {
    v3 = *(_QWORD *)(a1 + 8);
    result = *(_BYTE *)(v3 + 140) & 0xFB;
    if ( (*(_BYTE *)(v3 + 140) & 0xFB) == 8 )
    {
      result = sub_8D4C10(v3, dword_4F077C4 != 2);
      v4 = result;
      if ( (_DWORD)result )
      {
        v5 = sub_73D4C0(*(const __m128i **)(a1 + 8), dword_4F077C4 == 2);
        *(_QWORD *)(a1 + 8) = v5;
        if ( (v4 & 4) != 0 )
        {
          if ( unk_4F06908 )
            *(_QWORD *)(a1 + 8) = sub_73C570(v5, 4);
        }
        result = *(_DWORD *)(a1 + 32) & 0xFFFC07FF;
        *(_DWORD *)(a1 + 32) = result | ((v4 & 0x7F) << 11);
      }
    }
  }
  return result;
}
