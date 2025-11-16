// Function: sub_73E730
// Address: 0x73e730
//
_BYTE *__fastcall sub_73E730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rbx
  _BYTE *i; // r12
  __int64 *v10; // rax
  __int64 v12; // r13
  _QWORD *v13; // rax
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v6 = a1;
  v7 = *(_BYTE *)(a1 + 173);
  if ( v7 == 9 )
  {
LABEL_13:
    v12 = *(_QWORD *)(v6 + 176);
    i = 0;
    if ( (unsigned int)sub_731890(v12, 0, v14, a4, a5, a6) )
    {
      if ( *(_BYTE *)(v12 + 48) == 3 )
      {
        return *(_BYTE **)(v12 + 56);
      }
      else
      {
        v13 = sub_726700(5);
        v13[7] = v12;
        i = v13;
        *v13 = sub_73D720(*(const __m128i **)(v6 + 128));
      }
    }
  }
  else
  {
    while ( v7 != 10 )
    {
      if ( v7 != 11 )
        return 0;
      v6 = *(_QWORD *)(v6 + 176);
      v7 = *(_BYTE *)(v6 + 173);
      if ( v7 == 9 )
        goto LABEL_13;
    }
    v8 = *(_QWORD *)(v6 + 176);
    for ( i = 0; v8; v8 = *(_QWORD *)(v8 + 120) )
    {
      v10 = (__int64 *)sub_73E730(v8);
      if ( v10 )
      {
        if ( i )
          i = sub_73DF90((__int64)i, v10);
        else
          i = v10;
      }
    }
  }
  return i;
}
