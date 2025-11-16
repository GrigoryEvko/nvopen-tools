// Function: sub_3708E80
// Address: 0x3708e80
//
__int64 __fastcall sub_3708E80(__int64 a1)
{
  __int64 result; // rax
  int v2; // ebx
  char v3; // dl
  void (*v4)(void); // rax
  char v5; // [rsp-41h] [rbp-41h] BYREF
  __int64 v6; // [rsp-40h] [rbp-40h] BYREF

  result = *(_DWORD *)(a1 + 56) & 3;
  if ( (*(_DWORD *)(a1 + 56) & 3) != 0 )
  {
    v2 = 4 - result;
    do
    {
      v3 = v2 - 16;
      v4 = *(void (**)(void))(**(_QWORD **)(a1 + 24) + 16LL);
      if ( (char *)v4 != (char *)sub_3700C70 )
      {
        v4();
        v3 = v2 - 16;
      }
      v5 = v3;
      result = sub_3719260(&v6, a1, &v5, 1);
      if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      --v2;
    }
    while ( v2 );
  }
  return result;
}
