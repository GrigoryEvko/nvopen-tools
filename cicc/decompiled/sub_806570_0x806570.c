// Function: sub_806570
// Address: 0x806570
//
_QWORD *__fastcall sub_806570(__int64 a1, __m128i *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r13
  __int64 v4; // r12
  char v5; // dl
  _QWORD *result; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  int v9; // eax
  __int64 v10; // rcx
  char v11; // al
  int v12; // [rsp+4h] [rbp-3Ch]
  __int16 v13; // [rsp+8h] [rbp-38h]
  __int16 v14; // [rsp+Ah] [rbp-36h]
  int v15; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 48);
  v4 = *(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL);
  v5 = *(_BYTE *)(v4 + 176) & 0x10;
  if ( (*(_BYTE *)(v2 + 194) & 0x20) != 0 )
  {
    if ( v5 )
    {
      *(_BYTE *)(v2 + 205) = *(_BYTE *)(v2 + 205) & 0xE3 | 0x10;
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 89LL) & 8) != 0 )
        sub_80D2C0();
    }
    result = sub_801720(v3, *(_QWORD *)(a1 + 40), 0, 0, a2);
    *(_QWORD *)(a1 + 48) = 0;
  }
  else
  {
    v7 = 0;
    v8 = *(_QWORD *)(a1 + 40);
    v9 = dword_4D03F38[0];
    v10 = **(_QWORD **)(a1 + 80);
    *(_BYTE *)(v4 + 88) |= 4u;
    v15 = v9;
    v14 = dword_4D03F38[1];
    v12 = dword_4F07508[0];
    v13 = dword_4F07508[1];
    *(_QWORD *)dword_4D03F38 = v10;
    *(_QWORD *)dword_4F07508 = v10;
    if ( !v5 )
      goto LABEL_8;
    v7 = *(_QWORD *)(v8 + 112);
    if ( v3 )
    {
      while ( 1 )
      {
        v11 = *(_BYTE *)(v3 + 8);
        if ( v11 )
          break;
        v3 = *(_QWORD *)v3;
LABEL_8:
        if ( !v3 )
          goto LABEL_9;
      }
      while ( v11 == 1 )
      {
        sub_801720(v3, v8, 0, v7, a2);
        v3 = *(_QWORD *)v3;
        if ( !v3 )
          goto LABEL_9;
        v11 = *(_BYTE *)(v3 + 8);
      }
      sub_806430(v4, v8, v7, 0, a2->m128i_i32);
      do
      {
        sub_801720(v3, v8, 0, 0, a2);
        v3 = *(_QWORD *)v3;
      }
      while ( v3 );
    }
    else
    {
LABEL_9:
      sub_806430(v4, v8, v7, 0, a2->m128i_i32);
    }
    dword_4F07508[0] = v12;
    LOWORD(dword_4F07508[1]) = v13;
    dword_4D03F38[0] = v15;
    LOWORD(dword_4D03F38[1]) = v14;
    return dword_4D03F38;
  }
  return result;
}
