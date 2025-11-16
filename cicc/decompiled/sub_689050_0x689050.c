// Function: sub_689050
// Address: 0x689050
//
__int64 __fastcall sub_689050(_QWORD *a1, int a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  char v5; // dl
  __int64 v6; // rax
  unsigned int v7; // eax
  _DWORD v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v8[0] = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D3A70(*a1) )
    {
      sub_845C60(a1, 0, a2 == 0 ? 65 : 193, a2 == 0 ? 0x800 : 0, v8);
      result = v8[0];
      if ( v8[0] )
        goto LABEL_13;
    }
  }
  result = sub_6F69D0(a1, 0);
  if ( !*((_BYTE *)a1 + 16) )
    goto LABEL_13;
  v4 = *a1;
  v5 = *(_BYTE *)(*a1 + 140LL);
  if ( v5 == 12 )
  {
    result = *a1;
    do
    {
      result = *(_QWORD *)(result + 160);
      v5 = *(_BYTE *)(result + 140);
    }
    while ( v5 == 12 );
  }
  if ( !v5 || (result = sub_8D3D40(v4), (_DWORD)result) )
  {
LABEL_13:
    if ( !a2 )
      return result;
  }
  else
  {
    if ( !a2 )
      return sub_6E9350(a1);
    if ( !(unsigned int)sub_8D2930(*a1) )
    {
      v7 = sub_6E92D0();
      sub_6E68E0(v7, a1);
    }
  }
  if ( dword_4F077C4 != 1 )
    return sub_6FC420(a1);
  v6 = sub_72BA30(5);
  return sub_6FC3F0(v6, a1, 1);
}
