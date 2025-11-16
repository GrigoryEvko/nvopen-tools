// Function: sub_6B7AB0
// Address: 0x6b7ab0
//
__int64 __fastcall sub_6B7AB0(int a1, unsigned int a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // r12
  int v9; // [rsp+4h] [rbp-23Ch] BYREF
  __int64 v10; // [rsp+8h] [rbp-238h] BYREF
  _BYTE v11[160]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v12[2]; // [rsp+B0h] [rbp-190h] BYREF
  char v13; // [rsp+C1h] [rbp-17Fh]
  _BYTE v14[8]; // [rsp+F4h] [rbp-14Ch] BYREF
  __int64 v15; // [rsp+FCh] [rbp-144h]

  v9 = 0;
  sub_6E1DD0(&v10);
  sub_6E1E00(4, v11, 1, 0);
  sub_69ED20((__int64)v12, 0, 0, 0);
  if ( dword_4F077C4 == 2 && !a3 && (unsigned int)sub_8D3A70(v12[0]) )
    sub_845C60(v12, 0, 199, 2048, &v9);
  if ( !v9 )
    sub_6F69D0(v12, a1 != 0 ? 7 : 0);
  if ( a1 )
  {
    if ( dword_4F04C44 == -1
      && (v4 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v4 + 6) & 6) == 0)
      && *(_BYTE *)(v4 + 4) != 12
      || !(unsigned int)sub_8D3D40(v12[0]) )
    {
      v5 = v12[0];
      sub_6F9840(v12, 1, 1);
      if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v5) )
        sub_8AE000(v5);
      if ( v13 == 1 && !(unsigned int)sub_6ED0A0(v12) )
      {
        if ( (unsigned int)sub_8D2600(v5) )
        {
          if ( dword_4F077C0 )
          {
LABEL_20:
            sub_6E5C80(5, 137, v14);
LABEL_21:
            sub_6ECF90(v12, a2);
            goto LABEL_12;
          }
        }
        else if ( !(unsigned int)sub_8D23B0(v5) )
        {
          if ( (*(_BYTE *)(v5 + 140) & 0xFB) != 8 || (sub_8D4C10(v5, dword_4F077C4 != 2) & 1) == 0 )
          {
            if ( !(unsigned int)sub_8D3A70(v5) )
              goto LABEL_21;
            while ( *(_BYTE *)(v5 + 140) == 12 )
              v5 = *(_QWORD *)(v5 + 160);
            if ( (*(_BYTE *)(v5 + 176) & 2) == 0 )
              goto LABEL_21;
          }
          goto LABEL_20;
        }
      }
      sub_6E5C80(8, 137, v14);
      sub_6E6840(v12);
    }
  }
LABEL_12:
  v6 = sub_6F6F40(v12, 0);
  v7 = sub_6E2700(v6);
  sub_6E2B30(v6, 0);
  sub_6E1DF0(v10);
  unk_4F061D8 = v15;
  return v7;
}
