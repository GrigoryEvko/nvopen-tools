// Function: sub_18FD9D0
// Address: 0x18fd9d0
//
unsigned __int64 __fastcall sub_18FD9D0(int *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // rax
  unsigned __int64 v8; // rax
  int v9; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v10; // [rsp+4h] [rbp-ACh]
  __int64 v11; // [rsp+Ch] [rbp-A4h]
  __int64 v12; // [rsp+14h] [rbp-9Ch]
  __int64 v13; // [rsp+78h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v8 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v8 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v8;
    sub_2207640(byte_4F99930);
  }
  v9 = *a1;
  v6 = *a2;
  v13 = qword_4F99938;
  v10 = v6;
  v11 = *a3;
  v12 = *a4;
  return sub_1593600(&v9, 0x1Cu, qword_4F99938);
}
