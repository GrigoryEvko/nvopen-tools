// Function: sub_38CF4D0
// Address: 0x38cf4d0
//
bool __fastcall sub_38CF4D0(__int64 a1, __int64 a2)
{
  int v2; // edx
  bool result; // al
  __int64 v4; // r8
  int v5; // ecx
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // eax
  int v12; // r10d

  v2 = *(_DWORD *)(a1 + 176);
  result = 0;
  if ( v2 )
  {
    v4 = *(_QWORD *)(a2 + 24);
    v5 = v2 - 1;
    v6 = *(_QWORD *)(a1 + 160);
    v7 = (v2 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_3:
      v10 = v8[1];
      if ( v10 )
        return *(_DWORD *)(a2 + 20) <= *(_DWORD *)(v10 + 20);
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v12 = v11 + 1;
        v7 = v5 & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v4 == *v8 )
          goto LABEL_3;
        v11 = v12;
      }
    }
    return 0;
  }
  return result;
}
