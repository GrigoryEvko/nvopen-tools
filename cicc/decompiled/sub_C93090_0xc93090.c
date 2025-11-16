// Function: sub_C93090
// Address: 0xc93090
//
__int64 __fastcall sub_C93090(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // r10
  char v9; // dl
  char v10; // al

  v3 = a1[1];
  if ( a3 > v3 )
    return 0;
  v6 = v3 - a3;
  if ( a3 )
  {
    v7 = 0;
    v8 = *a1 + v6;
    while ( 1 )
    {
      v9 = *(_BYTE *)(v8 + v7);
      if ( (unsigned __int8)(v9 - 65) < 0x1Au )
        v9 += 32;
      v10 = *(_BYTE *)(a2 + v7);
      if ( (unsigned __int8)(v10 - 65) < 0x1Au )
        v10 += 32;
      if ( v9 != v10 )
        break;
      if ( a3 == ++v7 )
        return 1;
    }
    return 0;
  }
  return 1;
}
