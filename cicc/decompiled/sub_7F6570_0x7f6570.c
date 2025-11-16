// Function: sub_7F6570
// Address: 0x7f6570
//
__int64 *__fastcall sub_7F6570(__int64 a1, _QWORD *a2, int a3, int a4)
{
  __int64 *result; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  _QWORD *v10; // rdx
  int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-38h]

  result = sub_7E5340(a1);
  v8 = (__int64)result;
  if ( !a3 && (*(_BYTE *)(a1 + 194) & 0x10) != 0 )
    v8 = *result;
  if ( v8 )
  {
    while ( a3 || (*(_BYTE *)(v8 + 32) & 4) == 0 )
    {
      v9 = (__int64)sub_73C570(*(const __m128i **)(v8 + 8), (*(_DWORD *)(v8 + 32) >> 11) & 0x7F);
      if ( (*(_BYTE *)(v8 + 32) & 1) != 0 && (*(_BYTE *)(a1 + 198) & 0x20) == 0 && (*(_BYTE *)(v8 - 8) & 8) == 0 )
      {
        v13 = v9;
        if ( a4 )
        {
          sub_7E4BD0(v8);
          v9 = v13;
        }
        else
        {
          v9 = sub_7E4BA0(v8);
        }
      }
      v10 = sub_724EF0(v9);
      v11 = *(_BYTE *)(v8 + 32) & 4 | v10[4] & 0xFB;
      *((_BYTE *)v10 + 32) = v11;
      v12 = *(_BYTE *)(v8 + 32) & 8 | v11 & 0xFFFFFFF7;
      *((_BYTE *)v10 + 32) = v12;
      result = (__int64 *)(*(_BYTE *)(v8 + 32) & 1 | v12 & 0xFFFFFFFE);
      *((_BYTE *)v10 + 32) = (_BYTE)result;
      *a2 = v10;
      v8 = *(_QWORD *)v8;
      if ( !v8 )
        break;
      a2 = v10;
    }
  }
  return result;
}
