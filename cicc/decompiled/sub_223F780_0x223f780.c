// Function: sub_223F780
// Address: 0x223f780
//
void __fastcall sub_223F780(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A073F0;
  *(_QWORD *)(a1 + 16) = off_4A07418;
  *(_QWORD *)(a1 + 128) = off_4A07440;
  *(_QWORD *)(a1 + 24) = off_4A07080;
  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 != a1 + 112 )
    j___libc_free_0(v2);
  *(_QWORD *)(a1 + 24) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 80));
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = qword_4A07288;
  *(_QWORD *)a1 = qword_4A072D8;
  *(_QWORD *)(a1 + 128) = off_4A06798;
  sub_222E050(a1 + 128);
}
