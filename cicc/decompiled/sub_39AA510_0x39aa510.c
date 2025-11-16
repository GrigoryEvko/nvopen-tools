// Function: sub_39AA510
// Address: 0x39aa510
//
__int64 __fastcall sub_39AA510(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl
  unsigned int v3; // r8d
  __int64 v4; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // eax

  v1 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v1 )
  {
    v2 = 0;
    v3 = 0;
    v4 = 40 * v1;
    v6 = 0;
    while ( 1 )
    {
      v7 = v6 + *(_QWORD *)(a1 + 32);
      if ( *(_BYTE *)v7 == 10 )
      {
        v8 = *(_QWORD *)(v7 + 24);
        if ( !*(_BYTE *)(v8 + 16) )
        {
          if ( v2 )
            return 0;
          v9 = sub_1560180(v8 + 112, 30);
          v2 = 1;
          v3 = v9;
        }
      }
      v6 += 40;
      if ( v4 == v6 )
        return v3;
    }
  }
  return 0;
}
