// Function: sub_14AA3D0
// Address: 0x14aa3d0
//
__int64 __fastcall sub_14AA3D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // rdi
  int v9; // edx
  int v10; // r9d

  if ( !*(_BYTE *)(a1 + 184) )
    sub_14CDF70(a1);
  v3 = *(unsigned int *)(a1 + 176);
  if ( !(_DWORD)v3 )
    return 0;
  v5 = *(_QWORD *)(a1 + 160);
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = v5 + 88LL * v6;
  v8 = *(_QWORD *)(v7 + 24);
  if ( a2 != v8 )
  {
    v9 = 1;
    while ( v8 != -8 )
    {
      v10 = v9 + 1;
      v6 = (v3 - 1) & (v9 + v6);
      v7 = v5 + 88LL * v6;
      v8 = *(_QWORD *)(v7 + 24);
      if ( a2 == v8 )
        goto LABEL_6;
      v9 = v10;
    }
    return 0;
  }
LABEL_6:
  if ( v7 == v5 + 88 * v3 )
    return 0;
  return *(_QWORD *)(v7 + 40);
}
