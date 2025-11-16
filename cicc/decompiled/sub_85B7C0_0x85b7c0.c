// Function: sub_85B7C0
// Address: 0x85b7c0
//
__int64 __fastcall sub_85B7C0(_QWORD *a1, int a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 *v6; // rax

  v2 = qword_4F04C68[0] + 776LL * *(int *)(qword_4F04C68[0] + 776LL * a2 + 400);
  v3 = *(_QWORD *)(v2 + 688);
  if ( !v3 )
  {
    v6 = (__int64 *)qword_4F5FD20;
    if ( qword_4F5FD20 )
    {
      *(_QWORD *)(v2 + 688) = qword_4F5FD20;
      qword_4F5FD20 = *v6;
    }
    else
    {
      v6 = (__int64 *)sub_823970(128);
      *(_QWORD *)(v2 + 688) = v6;
    }
    *v6 = 0;
    v6[15] = 0;
    memset(
      (void *)((unsigned __int64)(v6 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v6 - (((_DWORD)v6 + 8) & 0xFFFFFFF8) + 128) >> 3));
    v3 = *(_QWORD *)(v2 + 688);
  }
  if ( (unsigned int)sub_879510(a1) )
    v4 = sub_85B780(a1);
  else
    v4 = *(_QWORD *)(*a1 + 8LL);
  return v3 + 8 * (sub_887620(v4) & 0xF);
}
