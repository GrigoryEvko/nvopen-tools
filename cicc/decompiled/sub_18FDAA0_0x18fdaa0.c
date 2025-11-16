// Function: sub_18FDAA0
// Address: 0x18fdaa0
//
unsigned __int64 __fastcall sub_18FDAA0(int *a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v4; // rax
  int v5; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v6; // [rsp+4h] [rbp-9Ch]
  __int64 v7; // [rsp+78h] [rbp-28h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v4 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v4 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v4;
    sub_2207640(byte_4F99930);
  }
  v5 = *a1;
  v2 = *a2;
  v7 = qword_4F99938;
  v6 = v2;
  return sub_1593600(&v5, 0xCu, qword_4F99938);
}
