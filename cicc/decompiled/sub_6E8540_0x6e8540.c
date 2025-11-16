// Function: sub_6E8540
// Address: 0x6e8540
//
__int64 __fastcall sub_6E8540(__int64 a1)
{
  char v2; // al
  __int64 v3; // rdi
  bool v4; // zf
  __int64 v6; // rdi
  __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_BYTE *)(a1 + 16);
  v7 = 0;
  if ( v2 == 1 )
  {
    v6 = *(_QWORD *)(a1 + 144);
LABEL_9:
    if ( !v6 )
    {
      v3 = *(_QWORD *)a1;
      v4 = *(_BYTE *)(a1 + 17) == 1;
      v7 = *(_QWORD *)a1;
      if ( !v4 )
        return sub_8D6540(v3);
      goto LABEL_11;
    }
    goto LABEL_6;
  }
  if ( v2 != 2 )
    goto LABEL_3;
  v6 = *(_QWORD *)(a1 + 288);
  if ( !v6 )
  {
    if ( *(_BYTE *)(a1 + 317) != 12 || *(_BYTE *)(a1 + 320) != 1 )
    {
LABEL_3:
      v3 = *(_QWORD *)a1;
      v4 = *(_BYTE *)(a1 + 17) == 1;
      v7 = *(_QWORD *)a1;
      if ( !v4 )
        return sub_8D6540(v3);
LABEL_11:
      v7 = sub_73D720(v3);
      v3 = v7;
      return sub_8D6540(v3);
    }
    v6 = sub_72E9A0(a1 + 144);
    goto LABEL_9;
  }
LABEL_6:
  if ( !(unsigned int)sub_6DF3C0(v6, &v7) )
    goto LABEL_3;
  return v7;
}
