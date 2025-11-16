// Function: sub_1E5E730
// Address: 0x1e5e730
//
__int64 *__fastcall sub_1E5E730(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v5; // edx

  v4 = sub_22077B0(56);
  if ( v4 )
  {
    *(_QWORD *)v4 = a2;
    v5 = 0;
    *(_QWORD *)(v4 + 8) = a3;
    if ( a3 )
      v5 = *(_DWORD *)(a3 + 16) + 1;
    *(_DWORD *)(v4 + 16) = v5;
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = -1;
  }
  *a1 = v4;
  return a1;
}
