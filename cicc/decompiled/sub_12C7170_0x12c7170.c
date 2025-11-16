// Function: sub_12C7170
// Address: 0x12c7170
//
__int64 __fastcall sub_12C7170(__int64 a1, const char *a2, const char *a3, const char *a4, __int64 a5)
{
  __int64 result; // rax
  size_t v8; // rax
  size_t v9; // rax
  size_t v10; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  result = a1 + 80;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = a5;
  if ( a2 )
  {
    v8 = strlen(a2);
    result = sub_2241130(a1, 0, 0, a2, v8);
  }
  if ( a3 )
  {
    v9 = strlen(a3);
    result = sub_2241130(a1 + 32, 0, *(_QWORD *)(a1 + 40), a3, v9);
  }
  if ( a4 )
  {
    v10 = strlen(a4);
    return sub_2241130(a1 + 64, 0, *(_QWORD *)(a1 + 72), a4, v10);
  }
  return result;
}
