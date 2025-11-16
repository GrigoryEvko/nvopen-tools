// Function: sub_8C0170
// Address: 0x8c0170
//
__int64 __fastcall sub_8C0170(_QWORD *a1, __int64 **a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // r14
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rbx
  __m128i *v9; // [rsp+8h] [rbp-28h] BYREF

  v9 = (__m128i *)a2;
  result = sub_72F130(a1);
  if ( result )
  {
    if ( (*(_BYTE *)(result + 207) & 0x10) != 0 )
    {
      v5 = **(_QWORD **)(result + 248);
      v9 = sub_72F240(v9);
      result = (__int64)sub_8B74F0(v5, (__int64 ***)&v9, 0, dword_4F07508, v6, v7);
      v8 = *(_QWORD *)(result + 88);
      if ( (*(_BYTE *)(v8 + 207) & 0x10) != 0 )
      {
        sub_8B1A30(*(_QWORD *)(result + 88), (FILE *)dword_4F07508);
        result = *(_QWORD *)(v8 + 152);
        if ( *(_BYTE *)(result + 140) == 7 )
        {
          *(_QWORD *)(a3 + 160) = *(_QWORD *)(result + 160);
          return sub_7325D0(a3, &dword_4F077C8);
        }
      }
    }
  }
  return result;
}
