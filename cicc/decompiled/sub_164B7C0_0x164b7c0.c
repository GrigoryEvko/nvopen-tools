// Function: sub_164B7C0
// Address: 0x164b7c0
//
void __fastcall sub_164B7C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  __int64 v12[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  v11 = 0;
  if ( (*(_BYTE *)(a1 + 23) & 0x20) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
      return;
    goto LABEL_14;
  }
  if ( (unsigned __int8)sub_1648C30(a1, &v11) )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
      goto LABEL_15;
  }
  else
  {
    v3 = v11;
    if ( v11 )
    {
      v4 = sub_16498B0(a1);
      sub_164D860(v3, v4);
    }
    sub_164B400(a1);
    if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
    {
      if ( v11 )
      {
LABEL_9:
        sub_1648C30(a2, v12);
        v5 = v12[0];
        if ( v11 == v12[0] )
        {
          v10 = sub_16498B0(a2);
          sub_164B0D0(a1, v10);
          sub_164B0D0(a2, 0);
          *(_QWORD *)(sub_16498B0(a1) + 8) = a1;
        }
        else
        {
          if ( v12[0] )
          {
            v6 = sub_16498B0(a2);
            sub_164D860(v5, v6);
          }
          v7 = sub_16498B0(a2);
          sub_164B0D0(a1, v7);
          sub_164B0D0(a2, 0);
          v8 = sub_16498B0(a1);
          v9 = v11;
          *(_QWORD *)(v8 + 8) = a1;
          if ( v9 )
            sub_164D6D0(v9, a1);
        }
        return;
      }
LABEL_14:
      if ( (unsigned __int8)sub_1648C30(a1, &v11) )
      {
LABEL_15:
        v13 = 257;
        sub_164B780(a2, v12);
        return;
      }
      goto LABEL_9;
    }
  }
}
