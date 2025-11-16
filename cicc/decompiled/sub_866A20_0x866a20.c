// Function: sub_866A20
// Address: 0x866a20
//
unsigned int *__fastcall sub_866A20(int a1, __int64 a2, char a3, unsigned int a4)
{
  if ( a1 )
    sub_866920(a2);
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                            & 0xFFFFFBFFFFFFFFEFLL
                                                            | (16LL * (a4 & 1))
                                                            | ((unsigned __int64)(a3 & 1) << 42);
  dword_4F04C3C = a4;
  return &dword_4F04C3C;
}
