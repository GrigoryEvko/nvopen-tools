// Function: sub_37F0C40
// Address: 0x37f0c40
//
bool __fastcall sub_37F0C40(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rbp
  __int64 v3; // rcx
  unsigned int v4; // edx
  __int64 v6; // rbx
  unsigned __int64 v7[5]; // [rsp-28h] [rbp-28h] BYREF

  v3 = *(unsigned __int16 *)(a2 + 6);
  v4 = *(_DWORD *)(a1[7] + 4 * v3);
  if ( !v4 || !(_WORD)v3 )
    return 0;
  v7[4] = v2;
  v6 = a1[1];
  v7[0] = a1[5];
  v7[1] = v4;
  return v6 + 8 != (_QWORD)sub_37F0BD0(v6, v7);
}
