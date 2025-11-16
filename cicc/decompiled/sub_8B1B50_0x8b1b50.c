// Function: sub_8B1B50
// Address: 0x8b1b50
//
__int64 __fastcall sub_8B1B50(_QWORD *a1, const __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // rax
  const __m128i *v8; // [rsp+8h] [rbp-28h] BYREF

  v8 = a2;
  result = sub_72F130(a1);
  if ( result )
  {
    if ( (*(_BYTE *)(result + 207) & 0x10) != 0 )
    {
      v5 = **(_QWORD **)(result + 248);
      v8 = sub_72F240(v8);
      result = sub_8B74F0(v5, &v8, 0, dword_4F07508);
      v6 = *(_QWORD *)(result + 88);
      if ( (*(_BYTE *)(v6 + 207) & 0x10) != 0 )
      {
        sub_8B1A30(*(_QWORD *)(result + 88), (FILE *)dword_4F07508);
        result = *(_QWORD *)(v6 + 152);
        if ( *(_BYTE *)(result + 140) == 7 )
        {
          v7 = sub_8D46C0(*(_QWORD *)(a3 + 160));
          *(_QWORD *)(v7 + 160) = *(_QWORD *)(*(_QWORD *)(v6 + 152) + 160LL);
          return sub_7325D0(v7, &dword_4F077C8);
        }
      }
    }
  }
  return result;
}
