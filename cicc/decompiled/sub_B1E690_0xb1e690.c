// Function: sub_B1E690
// Address: 0xb1e690
//
__int64 __fastcall sub_B1E690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v5 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(a3 + 32) || (result = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v4)) == 0 )
  {
    v7 = sub_B1E0B0(a1, a2);
    v8 = sub_B1E690(a1, *(_QWORD *)(v7 + 16), a3);
    return sub_B1B5D0(a3, a2, v8);
  }
  return result;
}
