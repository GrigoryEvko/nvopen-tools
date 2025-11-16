// Function: sub_3702B60
// Address: 0x3702b60
//
__int64 __fastcall sub_3702B60(__int64 a1)
{
  __int64 result; // rax
  int v2; // ebx
  char v3; // [rsp-31h] [rbp-31h] BYREF
  __int64 v4; // [rsp-30h] [rbp-30h] BYREF

  result = *(_DWORD *)(a1 + 56) & 3;
  if ( (*(_DWORD *)(a1 + 56) & 3) != 0 )
  {
    v2 = 4 - result;
    do
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 16LL))(*(_QWORD *)(a1 + 24));
      v3 = v2 - 16;
      result = sub_3719260(&v4, a1, &v3, 1);
      if ( (v4 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      --v2;
    }
    while ( v2 );
  }
  return result;
}
