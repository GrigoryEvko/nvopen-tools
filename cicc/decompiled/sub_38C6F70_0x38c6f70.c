// Function: sub_38C6F70
// Address: 0x38c6f70
//
unsigned __int64 __fastcall sub_38C6F70(__int64 *a1, int *a2, int *a3, char *a4, char *a5, int *a6)
{
  __int64 *v6; // r10
  int v11; // eax
  int v13; // eax
  unsigned __int64 v14; // rdx
  __int64 v15; // [rsp+10h] [rbp-B0h] BYREF
  int v16; // [rsp+18h] [rbp-A8h]
  int v17; // [rsp+1Ch] [rbp-A4h]
  char v18; // [rsp+20h] [rbp-A0h]
  char v19; // [rsp+21h] [rbp-9Fh]
  int v20; // [rsp+22h] [rbp-9Eh]
  __int64 v21; // [rsp+88h] [rbp-38h]

  v6 = a1;
  if ( !byte_4F99930[0] )
  {
    v13 = sub_2207590((__int64)byte_4F99930);
    v6 = a1;
    if ( v13 )
    {
      v14 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v14 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v14;
      sub_2207640((__int64)byte_4F99930);
      v6 = a1;
    }
  }
  v15 = *v6;
  v11 = *a2;
  v21 = qword_4F99938;
  v16 = v11;
  v17 = *a3;
  v18 = *a4;
  v19 = *a5;
  v20 = *a6;
  return sub_1593600(&v15, 0x16u, qword_4F99938);
}
