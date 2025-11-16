// Function: sub_18FD850
// Address: 0x18fd850
//
unsigned __int64 __fastcall sub_18FD850(int *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  unsigned __int64 v6; // rax
  int v7; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v8; // [rsp+4h] [rbp-ACh]
  __int64 v9; // [rsp+Ch] [rbp-A4h]
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
  return sub_1593600(&v7, 0x14u, qword_4F99938);
}
