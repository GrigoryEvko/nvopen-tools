// Function: sub_6301D0
// Address: 0x6301d0
//
__int64 __fastcall sub_6301D0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v6; // r12
  int v8; // r13d
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE v14[64]; // [rsp+10h] [rbp-B0h] BYREF
  char v15[112]; // [rsp+50h] [rbp-70h] BYREF

  v6 = a5;
  v8 = sub_8D3410(a5);
  if ( *(_BYTE *)(a2 + 80) == 7 )
  {
    if ( a4 )
    {
      if ( (*(_BYTE *)(a1 + 170) & 0x10) == 0 && dword_4F04C44 == -1 )
      {
        v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v11 + 6) & 6) == 0 && *(_BYTE *)(v11 + 4) != 12 )
        {
          sub_878710(a2, v14);
          if ( (v14[17] & 0x20) == 0 )
          {
            v12 = sub_879D20(v14, (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 88LL) >> 4) & 7, 0, 0, 0, v15);
            sub_6476B0(v12, a3, v6, 8);
          }
        }
      }
    }
  }
  if ( (unsigned int)sub_8D23B0(v6) )
  {
    if ( !v8 )
    {
LABEL_8:
      sub_6851C0(1345, a3);
LABEL_9:
      result = sub_72C930();
      v6 = result;
      goto LABEL_6;
    }
LABEL_13:
    v10 = sub_8D4050(v6);
    if ( (unsigned int)sub_8D97B0(v10) )
      goto LABEL_9;
    goto LABEL_8;
  }
  result = HIDWORD(qword_4F077B4);
  if ( !HIDWORD(qword_4F077B4) && v8 )
  {
    for ( result = v6; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
      ;
    if ( (*(_BYTE *)(result + 169) & 0x20) != 0 )
      goto LABEL_13;
  }
LABEL_6:
  *(_QWORD *)(a1 + 120) = v6;
  return result;
}
