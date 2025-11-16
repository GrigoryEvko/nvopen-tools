// Function: sub_16A4C70
// Address: 0x16a4c70
//
unsigned __int64 __fastcall sub_16A4C70(__int64 a1)
{
  char v1; // al
  char v2; // dl
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 *v5; // rax
  _DWORD *v6; // rdx
  char v7; // cl
  __int64 v9; // r15
  bool v10; // r12
  char v11; // bl
  unsigned __int64 v12; // rax
  char v13; // [rsp+Eh] [rbp-B2h] BYREF
  char v14; // [rsp+Fh] [rbp-B1h] BYREF
  _QWORD v15[22]; // [rsp+10h] [rbp-B0h] BYREF

  v1 = *(_BYTE *)(a1 + 18);
  v2 = v1 & 7;
  if ( (v1 & 7) == 1 )
  {
    v9 = *(_QWORD *)a1;
    v10 = 0;
  }
  else
  {
    if ( v2 != 3 && v2 )
    {
      v3 = sub_16984A0(a1);
      v4 = v3 + 8LL * (unsigned int)sub_1698310(a1);
      v5 = (__int64 *)sub_16984A0(a1);
      v15[0] = sub_15B1DB0(v5, v4);
      v6 = (_DWORD *)(*(_QWORD *)a1 + 4LL);
      v7 = *(_BYTE *)(a1 + 18) >> 3;
      v14 = *(_BYTE *)(a1 + 18) & 7;
      v13 = v7 & 1;
      return sub_16A3620(&v14, &v13, v6, (_WORD *)(a1 + 16), v15);
    }
    v9 = *(_QWORD *)a1;
    v10 = (v1 & 8) != 0;
  }
  v11 = v1 & 7;
  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v12 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v12 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v12;
    sub_2207640(byte_4F99930);
  }
  LOBYTE(v15[0]) = v11;
  BYTE1(v15[0]) = v10;
  v15[15] = qword_4F99938;
  *(_DWORD *)((char *)v15 + 2) = *(_DWORD *)(v9 + 4);
  return sub_1593600(v15, 6u, qword_4F99938);
}
