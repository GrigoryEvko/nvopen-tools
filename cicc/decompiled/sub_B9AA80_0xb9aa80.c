// Function: sub_B9AA80
// Address: 0xb9aa80
//
void __fastcall sub_B9AA80(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx

  *(_DWORD *)(a2 + 8) = 0;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 )
  {
    v3 = 0;
    if ( !*(_DWORD *)(a2 + 12) )
    {
      sub_C8D5F0(a2, a2 + 16, 1, 16);
      v3 = 16LL * *(unsigned int *)(a2 + 8);
    }
    v4 = *(_QWORD *)a2;
    *(_QWORD *)(v4 + v3) = 0;
    *(_QWORD *)(v4 + v3 + 8) = v2;
    ++*(_DWORD *)(a2 + 8);
  }
  sub_B9A9D0(a1, a2);
}
