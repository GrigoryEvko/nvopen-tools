// Function: sub_1E35590
// Address: 0x1e35590
//
unsigned __int64 __fastcall sub_1E35590(char *a1, int *a2, int *a3, char *a4)
{
  unsigned __int64 v7; // rax
  char src; // [rsp+0h] [rbp-B0h] BYREF
  int v9; // [rsp+1h] [rbp-AFh]
  int v10; // [rsp+5h] [rbp-ABh]
  char v11; // [rsp+9h] [rbp-A7h]
  char v12[54]; // [rsp+Ah] [rbp-A6h] BYREF
  char v13[56]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v14; // [rsp+78h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v7 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v7 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v7;
    sub_2207640(byte_4F99930);
  }
  v14 = qword_4F99938;
  src = *a1;
  v9 = *a2;
  v10 = *a3;
  v11 = *a4;
  return sub_1E30AD0(&src, 0, v12, v13);
}
