// Function: sub_253B250
// Address: 0x253b250
//
__int64 __fastcall sub_253B250(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx
  __int64 v4; // r8
  int v5; // ecx
  unsigned int v6; // edx
  __int64 v7; // rdi
  int v8; // eax

  result = 0;
  if ( *(_BYTE *)(a1 + 96) && *(_BYTE *)(a1 + 97) )
  {
    v3 = *(_DWORD *)(a1 + 384);
    v4 = *(_QWORD *)(a1 + 368);
    if ( v3 )
    {
      v5 = v3 - 1;
      v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = *(_QWORD *)(v4 + 8LL * v6);
      if ( a2 == v7 )
        return result;
      v8 = 1;
      while ( v7 != -4096 )
      {
        v6 = v5 & (v8 + v6);
        v7 = *(_QWORD *)(v4 + 8LL * v6);
        if ( a2 == v7 )
          return 0;
        ++v8;
      }
    }
    return 1;
  }
  return result;
}
