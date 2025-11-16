// Function: sub_215BB50
// Address: 0x215bb50
//
__int64 __fastcall sub_215BB50(__int64 a1)
{
  strcpy((char *)(a1 + 16), "%ERROR");
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 6;
  return a1;
}
