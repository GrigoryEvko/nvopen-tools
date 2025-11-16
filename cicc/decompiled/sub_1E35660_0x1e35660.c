// Function: sub_1E35660
// Address: 0x1e35660
//
unsigned __int64 __fastcall sub_1E35660(char *a1, int *a2, __int64 *a3)
{
  unsigned __int64 v5; // rax
  char src; // [rsp+0h] [rbp-B0h] BYREF
  int v7; // [rsp+1h] [rbp-AFh]
  __int64 v8; // [rsp+5h] [rbp-ABh]
  char v9[51]; // [rsp+Dh] [rbp-A3h] BYREF
  char v10[56]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v11; // [rsp+78h] [rbp-38h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v5 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v5 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v5;
    sub_2207640(byte_4F99930);
  }
  v11 = qword_4F99938;
  src = *a1;
  v7 = *a2;
  v8 = *a3;
  return sub_1E30AD0(&src, 0, v9, v10);
}
