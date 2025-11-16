// Function: sub_2A64F10
// Address: 0x2a64f10
//
__int64 __fastcall sub_2A64F10(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  __int64 v3; // rdi
  unsigned int v4; // edx
  __int64 *v5; // rax
  __int64 v6; // r8
  int v8; // eax
  int v9; // r10d

  v2 = *(_DWORD *)(*(_QWORD *)a1 + 160LL);
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 144LL);
  if ( v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 48LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
      return (__int64)(v5 + 1);
    v8 = 1;
    while ( v6 != -4096 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 48LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        return (__int64)(v5 + 1);
      v8 = v9;
    }
  }
  return v3 + 48LL * v2 + 8;
}
