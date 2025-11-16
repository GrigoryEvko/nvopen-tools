// Function: sub_16A3620
// Address: 0x16a3620
//
unsigned __int64 __fastcall sub_16A3620(_BYTE *a1, char *a2, _DWORD *a3, _WORD *a4, _QWORD *a5)
{
  char v8; // al
  unsigned __int64 v10; // rdx
  _QWORD v11[22]; // [rsp+10h] [rbp-B0h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v10 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v10 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v10;
    sub_2207640(byte_4F99930);
  }
  LOBYTE(v11[0]) = *a1;
  v8 = *a2;
  v11[15] = qword_4F99938;
  BYTE1(v11[0]) = v8;
  *(_DWORD *)((char *)v11 + 2) = *a3;
  HIWORD(v11[0]) = *a4;
  v11[1] = *a5;
  return sub_1593600(v11, 0x10u, qword_4F99938);
}
