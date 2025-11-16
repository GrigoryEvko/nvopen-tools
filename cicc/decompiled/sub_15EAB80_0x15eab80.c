// Function: sub_15EAB80
// Address: 0x15eab80
//
unsigned __int64 __fastcall sub_15EAB80(__int64 *a1, __int64 *a2, char *a3, char *a4, int *a5, __int64 *a6)
{
  char *v6; // r10
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  int v18; // eax
  unsigned __int64 v19; // rcx
  char *v20; // [rsp+0h] [rbp-C0h]
  _QWORD v22[2]; // [rsp+10h] [rbp-B0h] BYREF
  char v23; // [rsp+20h] [rbp-A0h]
  char v24; // [rsp+21h] [rbp-9Fh]
  int v25; // [rsp+22h] [rbp-9Eh]
  __int64 v26; // [rsp+26h] [rbp-9Ah]
  __int64 v27; // [rsp+88h] [rbp-38h]

  v6 = a3;
  if ( !byte_4F99930[0] )
  {
    v18 = sub_2207590(byte_4F99930);
    v6 = a3;
    if ( v18 )
    {
      v19 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v19 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v19;
      sub_2207640(byte_4F99930);
      v6 = a3;
    }
  }
  v12 = *a1;
  v20 = v6;
  v13 = a1[1];
  v27 = qword_4F99938;
  v14 = sub_16D3930(v12, v13);
  v15 = *a2;
  v16 = a2[1];
  v22[0] = v14;
  v22[1] = sub_16D3930(v15, v16);
  v23 = *v20;
  v24 = *a4;
  v25 = *a5;
  v26 = *a6;
  return sub_1593600(v22, 0x1Eu, v27);
}
