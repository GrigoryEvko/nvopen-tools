// Function: sub_234BCC0
// Address: 0x234bcc0
//
__int64 __fastcall sub_234BCC0(__int64 a1, char *a2, char a3)
{
  char v4; // r13
  __int64 v5; // rax

  v4 = *a2;
  v5 = sub_22077B0(0x10u);
  if ( v5 )
  {
    *(_BYTE *)(v5 + 8) = v4;
    *(_QWORD *)v5 = &unk_4A10DF8;
  }
  *(_QWORD *)a1 = v5;
  *(_BYTE *)(a1 + 8) = a3;
  return a1;
}
