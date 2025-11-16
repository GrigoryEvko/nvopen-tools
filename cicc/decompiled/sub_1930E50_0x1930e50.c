// Function: sub_1930E50
// Address: 0x1930e50
//
unsigned __int64 __fastcall sub_1930E50(__int64 *a1, int *a2, char *a3)
{
  int v4; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // [rsp+0h] [rbp-B0h] BYREF
  int v8; // [rsp+8h] [rbp-A8h]
  char v9; // [rsp+Ch] [rbp-A4h]
  __int64 v10; // [rsp+78h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v6 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v6 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v6;
    sub_2207640(byte_4F99930);
  }
  v7 = *a1;
  v4 = *a2;
  v10 = qword_4F99938;
  v8 = v4;
  v9 = *a3;
  return sub_1593600(&v7, 0xDu, qword_4F99938);
}
