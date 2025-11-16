// Function: sub_73DE50
// Address: 0x73de50
//
_BYTE *__fastcall sub_73DE50(__int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 v4; // r15
  unsigned __int8 v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // rdx
  int v8; // esi
  __m128i *v9; // rax
  _BYTE *result; // rax
  char v11; // al
  __int64 *v12; // r14
  __int64 v13; // rax

  v3 = (__int64 *)a1;
  if ( !dword_4D03F94 || *(_BYTE *)(a1 + 24) != 1 || (*(_BYTE *)(a1 + 27) & 2) == 0 )
    goto LABEL_3;
  v11 = *(_BYTE *)(a1 + 56);
  v12 = *(__int64 **)(a1 + 72);
  if ( !v11 )
  {
    if ( (unsigned int)sub_8D3A70(*v12) )
    {
      v3 = v12;
      goto LABEL_3;
    }
    v11 = *(_BYTE *)(a1 + 56);
  }
  if ( v11 == 3 )
  {
    if ( (unsigned int)sub_8D2E30(*v12) )
    {
      v13 = sub_8D46C0(*v12);
      if ( (unsigned int)sub_8D3A70(v13) )
        v3 = v12;
    }
  }
LABEL_3:
  if ( (unsigned int)sub_8D2E30(*v3) )
  {
    v5 = 95;
    v4 = sub_8D46C0(*v3);
  }
  else
  {
    v4 = *v3;
    v5 = 94;
  }
  v6 = sub_726700(4);
  v7 = *(_QWORD *)(a2 + 120);
  v8 = 0;
  v6[7] = a2;
  *v6 = v7;
  v3[2] = (__int64)v6;
  if ( (*(_BYTE *)(v4 + 140) & 0xFB) == 8 )
    v8 = sub_8D4C10(v4, dword_4F077C4 != 2);
  v9 = sub_73CB50(a2, v8);
  result = sub_73DC30(v5, (__int64)v9, (__int64)v3);
  result[27] |= 2u;
  return result;
}
