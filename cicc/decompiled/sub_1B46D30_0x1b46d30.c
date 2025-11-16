// Function: sub_1B46D30
// Address: 0x1b46d30
//
__int64 __fastcall sub_1B46D30(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rbx
  int v4; // r14d
  unsigned __int64 v5; // rax

  v2 = sub_157EBA0(a2);
  v3 = v2;
  if ( v2 )
  {
    v4 = sub_15F4D60(v2);
    v5 = sub_157EBA0(a2);
  }
  else
  {
    v5 = 0;
    v4 = 0;
  }
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 16) = v3;
  *(_DWORD *)(a1 + 24) = v4;
  *(_DWORD *)(a1 + 8) = 0;
  return a1;
}
