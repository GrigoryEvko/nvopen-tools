// Function: sub_6AC060
// Address: 0x6ac060
//
__int64 __fastcall sub_6AC060(unsigned int a1, unsigned int a2, _DWORD *a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  char i; // dl
  int v8; // ecx
  int v9; // eax
  int v10; // ecx
  int v11; // [rsp+Ch] [rbp-194h]
  _QWORD v12[2]; // [rsp+10h] [rbp-190h] BYREF
  char v13; // [rsp+20h] [rbp-180h]
  char v14; // [rsp+21h] [rbp-17Fh]

  v4 = unk_4D04980;
  if ( unk_4D0455C && !unk_4D04980 )
  {
    sub_88A1F0();
    v4 = unk_4D04980;
  }
  if ( (unsigned int)sub_8D3410(v4) )
  {
    v4 = sub_8D67C0(v4);
    sub_69ED20((__int64)v12, 0, 0, 1);
    sub_6F69D0(v12, 0);
    if ( v14 != 2 )
      goto LABEL_6;
    v8 = 1;
  }
  else
  {
    sub_69ED20((__int64)v12, 0, 0, 1);
    sub_6F69D0(v12, 6);
    if ( v14 != 1 )
      goto LABEL_6;
    v8 = sub_6ED0A0(v12);
    if ( v8 )
      goto LABEL_6;
  }
  v11 = v8;
  v9 = sub_8DBE70(v12[0]);
  v10 = v11;
  if ( !v9 && v12[0] != v4 )
  {
    if ( (unsigned int)sub_8DED30(v4, v12[0], 3) )
    {
      v10 = v11;
      goto LABEL_19;
    }
LABEL_6:
    if ( v13 )
    {
      v5 = v12[0];
      for ( i = *(_BYTE *)(v12[0] + 140LL); i == 12; i = *(_BYTE *)(v5 + 140) )
        v5 = *(_QWORD *)(v5 + 160);
      if ( i )
        sub_6E68E0(a2, v12);
    }
    *a3 = 1;
    goto LABEL_12;
  }
LABEL_19:
  if ( *a3 )
  {
LABEL_12:
    sub_6E6450(v12);
    return 0;
  }
  if ( !v10 )
    sub_6ECF90(v12, a1);
  return sub_6F6F40(v12, 0);
}
