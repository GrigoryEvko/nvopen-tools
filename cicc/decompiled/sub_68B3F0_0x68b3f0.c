// Function: sub_68B3F0
// Address: 0x68b3f0
//
__int64 __fastcall sub_68B3F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // r14
  __int64 i; // rax
  __int64 v6; // rdx
  __int64 j; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // r8
  _BYTE v11[4]; // [rsp+14h] [rbp-2Ch] BYREF
  _BYTE v12[40]; // [rsp+18h] [rbp-28h] BYREF

  if ( dword_4F077BC
    && qword_4F077A8 <= 0x76BFu
    && *((_BYTE *)a1 + 17) == 1
    && (LODWORD(v2) = sub_6ED0A0(a1), !(_DWORD)v2)
    && *(_BYTE *)(unk_4D03C50 + 16LL) > 3u )
  {
    v4 = *a1;
    if ( (unsigned int)sub_8D2930(*a1) && (unsigned int)sub_8D2930(a2) && !(unsigned int)sub_8D29A0(a2)
      || (unsigned int)sub_8D2E30(v4)
      && (unsigned int)sub_8D2E30(a2)
      && (dword_4F077C4 != 2
       || !(unsigned int)sub_8D2E30(v4)
       || !(unsigned int)sub_8D2E30(a2)
       || !(unsigned int)sub_8D5EF0(v4, a2, v11, v12)) )
    {
      for ( i = v4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v6 = *(_QWORD *)(i + 128);
      for ( j = a2; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( v6 == *(_QWORD *)(j + 128) )
      {
        v8 = sub_73D720(a2);
        if ( !(unsigned int)sub_8D97D0(v4, v8, 0, v9, v10) )
          return (unsigned int)sub_6ECD10(a1) == 0;
      }
    }
  }
  else
  {
    LODWORD(v2) = 0;
  }
  return (unsigned int)v2;
}
