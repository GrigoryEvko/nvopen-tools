// Function: sub_1597150
// Address: 0x1597150
//
unsigned __int64 __fastcall sub_1597150(_BYTE *a1, char *a2, __int16 *a3, __int64 *a4, __int64 *a5)
{
  char v8; // al
  unsigned __int64 v10; // rdx
  _BYTE v11[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v12; // [rsp+12h] [rbp-AEh]
  __int64 v13; // [rsp+14h] [rbp-ACh]
  __int64 v14; // [rsp+1Ch] [rbp-A4h]
  __int64 v15; // [rsp+88h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v10 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v10 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v10;
    sub_2207640(byte_4F99930);
  }
  v11[0] = *a1;
  v8 = *a2;
  v15 = qword_4F99938;
  v11[1] = v8;
  v12 = *a3;
  v13 = *a4;
  v14 = *a5;
  return sub_1593600(v11, 0x14u, qword_4F99938);
}
