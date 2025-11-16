// Function: sub_83E620
// Address: 0x83e620
//
__int64 __fastcall sub_83E620(_BYTE *a1, const __m128i *a2)
{
  int v4; // r8d
  __int64 result; // rax
  _BOOL4 v6; // r14d
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // [rsp+4h] [rbp-3Ch] BYREF
  _DWORD v11[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  v4 = sub_8D23B0(a2);
  result = 0;
  if ( !v4 )
  {
    v6 = 1;
    if ( a1[17] != 2 )
      v6 = sub_6ED0A0((__int64)a1);
    v7 = *(_QWORD *)a1;
    v8 = 0;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 140LL) & 0xFB) == 8 )
      v8 = (unsigned int)sub_8D4C10(v7, dword_4F077C4 != 2);
    v9 = sub_83DE00(a2, v8, v6, 0, (__int64)(a1 + 68), (_DWORD *)&v10 + 1, v11, 0, &v10);
    result = HIDWORD(v10) | (unsigned int)v10;
    if ( v10 )
    {
      return 1;
    }
    else if ( v9 )
    {
      return v11[0] == 0;
    }
  }
  return result;
}
