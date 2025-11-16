// Function: sub_2EB6120
// Address: 0x2eb6120
//
__int64 __fastcall sub_2EB6120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  if ( a2 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v8 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  if ( v8 >= *(_DWORD *)(a3 + 56) || (result = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v7)) == 0 )
  {
    v10 = sub_2EB5B40(a1, a2, v7, a4, a5, a6);
    v11 = sub_2EB6120(a1, *(_QWORD *)(v10 + 16), a3);
    return sub_2EB4C20(a3, a2, v11);
  }
  return result;
}
