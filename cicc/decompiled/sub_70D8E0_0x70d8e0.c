// Function: sub_70D8E0
// Address: 0x70d8e0
//
__int64 __fastcall sub_70D8E0(__int64 a1, __m128i *a2, _DWORD *a3)
{
  bool v3; // zf
  __int64 *v4; // r15
  unsigned int v6; // r14d
  unsigned __int8 v7; // al
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 i; // rax
  const __m128i *v13; // rcx
  _BOOL4 v14; // [rsp+Ch] [rbp-44h] BYREF
  __int16 v15[32]; // [rsp+10h] [rbp-40h] BYREF

  v3 = *(_BYTE *)(a1 + 24) == 2;
  v14 = 0;
  if ( v3 )
    return 1;
  v4 = *(__int64 **)(a1 + 72);
  v6 = sub_70D8E0(v4);
  v7 = *(_BYTE *)(a1 + 56);
  if ( v7 == 92 )
  {
    for ( i = sub_8D46C0(*v4); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v13 = *(const __m128i **)(v4[2] + 56);
    v14 = 0;
    if ( v13[10].m128i_i8[13] != 1 )
    {
      if ( a3 )
      {
        v6 = 0;
        sub_6851C0(0x6F6u, a3);
        return v6;
      }
      return 0;
    }
    sub_70D820(a2, 0, 0, v13, *(_QWORD *)(i + 128), 0, &v14);
  }
  else
  {
    if ( v7 > 0x5Cu )
    {
      if ( (unsigned __int8)(v7 - 94) <= 1u )
      {
        sub_70CFA0((__int64)a2, *(_QWORD *)(v4[2] + 56), 0, &v14);
        if ( !v6 )
          return 0;
        goto LABEL_12;
      }
      goto LABEL_16;
    }
    if ( v7 == 14 )
    {
      v9 = *v4;
      v10 = *(_QWORD *)a1;
      if ( (unsigned int)sub_8D2E30(v9) )
      {
        v9 = sub_8D46C0(v9);
        v10 = sub_8D46C0(v10);
      }
      v11 = sub_8D5CE0(v9, v10);
      if ( (*(_BYTE *)(v11 + 96) & 2) != 0 )
      {
        if ( a3 )
          sub_6851C0(0x592u, a3);
        return 0;
      }
      sub_620DE0(v15, *(_QWORD *)(v11 + 104));
      sub_621270((unsigned __int16 *)&a2[11], v15, 0, &v14);
    }
    else
    {
      if ( v7 > 0xEu )
      {
        if ( v7 == 21 )
          goto LABEL_7;
LABEL_16:
        sub_721090(v4);
      }
      if ( v7 && v7 != 3 )
        goto LABEL_16;
    }
  }
LABEL_7:
  if ( !v6 )
    return 0;
LABEL_12:
  if ( v14 )
  {
    v6 = 0;
    sub_6851C0(0x4E7u, a3);
  }
  return v6;
}
