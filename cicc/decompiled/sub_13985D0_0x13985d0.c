// Function: sub_13985D0
// Address: 0x13985d0
//
__int64 __fastcall sub_13985D0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 + 8); *(_QWORD *)(i + 24) != a2 || *(_QWORD *)(i + 16); i += 32 )
    ;
  --*(_DWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD *)(i + 16);
  v6 = *(_QWORD *)(v4 - 16);
  if ( v5 != v6 )
  {
    if ( v5 != 0 && v5 != -8 && v5 != -16 )
    {
      sub_1649B30(i);
      v6 = *(_QWORD *)(v4 - 16);
    }
    *(_QWORD *)(i + 16) = v6;
    if ( v6 != 0 && v6 != -8 && v6 != -16 )
      sub_1649AC0(i, *(_QWORD *)(v4 - 32) & 0xFFFFFFFFFFFFFFF8LL);
  }
  *(_QWORD *)(i + 24) = *(_QWORD *)(v4 - 8);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = v7 - 32;
  *(_QWORD *)(a1 + 16) = v7 - 32;
  result = *(_QWORD *)(v7 - 16);
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v8);
  return result;
}
