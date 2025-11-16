// Function: sub_26C20D0
// Address: 0x26c20d0
//
unsigned __int64 __fastcall sub_26C20D0(__int64 a1, const char *a2, _BYTE *a3, int **a4, __int64 *a5)
{
  size_t v8; // rax
  __int64 v9; // rdx
  int v10; // eax

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
  v9 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v10 = **a4;
  *(_QWORD *)(a1 + 40) = v9;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v10;
  *(_DWORD *)(a1 + 152) = v10;
  *(_QWORD *)(a1 + 48) = a5[1];
  return sub_C53130(a1);
}
