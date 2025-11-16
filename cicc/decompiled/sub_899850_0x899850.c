// Function: sub_899850
// Address: 0x899850
//
__int64 __fastcall sub_899850(__int64 a1, FILE *a2)
{
  __int64 result; // rax
  __int64 v3; // r8

  if ( dword_4F077C4 != 2
    || (result = (__int64)&unk_4F07778, unk_4F07778 <= 201102) && (result = (__int64)&dword_4F07774, !dword_4F07774) )
  {
    result = (unsigned int)dword_4D04964;
    if ( dword_4D04964 )
    {
      result = sub_880920(a1);
      if ( !(_DWORD)result )
      {
        v3 = sub_8807C0(a1);
        result = qword_4F04C68[0] + 776LL * dword_4F04C34;
        if ( *(_QWORD *)(result + 224) != v3 )
          return sub_6853B0(unk_4F07471, 0x4B4u, a2, a1);
      }
    }
  }
  return result;
}
