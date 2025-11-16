// Function: sub_2ABC530
// Address: 0x2abc530
//
unsigned __int64 __fastcall sub_2ABC530(__int64 a1, const char *a2, int **a3, _BYTE *a4, __int64 *a5)
{
  size_t v8; // rax
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax

  sub_D95050(a1, 0, 0);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)(a1 + 144) = &unk_49D9728;
  *(_QWORD *)a1 = &unk_49DBF10;
  *(_QWORD *)(a1 + 160) = &unk_49DC290;
  *(_QWORD *)(a1 + 192) = nullsub_24;
  *(_QWORD *)(a1 + 184) = sub_984050;
  v8 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v8);
  v9 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v9;
  *(_DWORD *)(a1 + 152) = v9;
  v10 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v11 = a5[1];
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 48) = v11;
  return sub_C53130(a1);
}
