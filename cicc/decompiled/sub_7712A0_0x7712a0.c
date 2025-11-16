// Function: sub_7712A0
// Address: 0x7712a0
//
unsigned __int64 __fastcall sub_7712A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int64 v3; // rdx
  __int64 v5; // r8
  int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // edx
  unsigned __int64 result; // rax
  __int64 v10; // r9

  v2 = a1 + 72;
  v3 = (unsigned __int64)(a1 + 72) >> 3;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_DWORD *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v8 = v6 & v3;
  result = v7 + 16LL * v8;
  v10 = *(_QWORD *)result;
  if ( v5 )
  {
    if ( v2 != v10 )
    {
      do
      {
        v8 = v6 & (v8 + 1);
        result = v7 + 16LL * v8;
      }
      while ( *(_QWORD *)result != v2 );
    }
    *(_QWORD *)(result + 8) = v5;
  }
  else
  {
    if ( v2 != v10 )
    {
      do
      {
        v8 = v6 & (v8 + 1);
        result = v7 + 16LL * v8;
      }
      while ( *(_QWORD *)result != v2 );
    }
    *(_QWORD *)result = 0;
    result = v8 + 1;
    if ( *(_QWORD *)(v7 + 16LL * ((unsigned int)result & v6)) )
      result = sub_771200(*(_QWORD *)a1, *(_DWORD *)(a1 + 8), v8);
    --*(_DWORD *)(a1 + 12);
  }
  return result;
}
