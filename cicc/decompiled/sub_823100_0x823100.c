// Function: sub_823100
// Address: 0x823100
//
void __fastcall sub_823100(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  v6 = dword_4F073AC;
  if ( dword_4F073AC <= a1 )
  {
    dword_4F073AC = a1 + 2048;
    v7 = 8LL * v6;
    qword_4F073B0 = (void *)sub_822C60(qword_4F073B0, v7, 8LL * (a1 + 2048), a4, a5, a6);
    memset((char *)qword_4F073B0 + v7, 0, 8LL * (dword_4F073AC - v6));
    qword_4F072B0 = (void *)sub_822C60(qword_4F072B0, v7, 8LL * dword_4F073AC, v8, v9, v10);
    memset((char *)qword_4F072B0 + v7, 0, 8LL * (dword_4F073AC - v6));
  }
}
