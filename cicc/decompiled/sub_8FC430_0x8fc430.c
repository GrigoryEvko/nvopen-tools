// Function: sub_8FC430
// Address: 0x8fc430
//
__int64 __fastcall sub_8FC430(__int64 a1, const char *a2)
{
  size_t v2; // rax

  v2 = strlen(a2);
  return sub_2241130(a1, 0, *(_QWORD *)(a1 + 8), a2, v2);
}
