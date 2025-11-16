// Function: sub_1E34230
// Address: 0x1e34230
//
__int64 __fastcall sub_1E34230(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx

  v2 = sub_1EB3840(*(_QWORD *)(a2 + 376));
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v2 | 4;
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(v2 + 12);
  *(_DWORD *)(a1 + 20) = v3;
  return a1;
}
