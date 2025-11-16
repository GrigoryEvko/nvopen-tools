// Function: sub_23A1570
// Address: 0x23a1570
//
unsigned __int64 __fastcall sub_23A1570(__int64 a1, const char *a2, char **a3, _BYTE *a4, __int64 *a5)
{
  size_t v8; // rax
  char *v9; // rax
  char v10; // dl
  __int64 v11; // rdx
  __int64 v12; // rax

  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 136) = 0;
  *(_WORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9748;
  *(_QWORD *)a1 = &unk_49DC090;
  *(_QWORD *)(a1 + 160) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 192) = nullsub_23;
  *(_QWORD *)(a1 + 184) = sub_984030;
  v8 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v8);
  v9 = *a3;
  v10 = **a3;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 136) = v10;
  *(_BYTE *)(a1 + 152) = *v9;
  v11 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = a5[1];
  *(_QWORD *)(a1 + 40) = v11;
  *(_QWORD *)(a1 + 48) = v12;
  return sub_C53130(a1);
}
