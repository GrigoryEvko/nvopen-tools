// Function: sub_823040
// Address: 0x823040
//
void __fastcall sub_823040(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx

  v6 = dword_4F073A4;
  if ( dword_4F073A4 <= a1 )
  {
    dword_4F073A4 = a1 + 2048;
    qword_4F072B8 = (void *)sub_822C60(qword_4F072B8, 16LL * v6, 16LL * (a1 + 2048), a4, a5, a6);
    memset((char *)qword_4F072B8 + 16 * v6, 0, 16LL * (dword_4F073A4 - v6));
  }
}
