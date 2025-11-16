// Function: sub_866880
// Address: 0x866880
//
__int64 __fastcall sub_866880(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = dword_4F04C64;
  if ( dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
  {
    if ( dword_4F04C64 != -1 && (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 7) & 1) != 0) )
    {
      v3 = sub_85B260(a1, a2, 0);
      *(_BYTE *)(v3 + 42) = 1;
      *(_QWORD *)v3 = qword_4F04C18;
      qword_4F04C18 = (_QWORD *)v3;
      *a1 = v3;
      return (__int64)&qword_4F04C18;
    }
    else
    {
      *a1 = 0;
    }
  }
  else
  {
    *a1 = 0;
  }
  return result;
}
