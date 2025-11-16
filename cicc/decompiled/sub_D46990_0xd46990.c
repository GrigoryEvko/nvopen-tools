// Function: sub_D46990
// Address: 0xd46990
//
__int64 __fastcall sub_D46990(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  char v3; // cl
  __int64 v4; // rax
  __int64 v5; // rdi
  int v6; // eax
  int v7; // r9d
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r10
  int v12; // eax
  int v13; // r11d

  v2 = a1[1];
  v3 = *(_BYTE *)a1[2];
  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 8);
  v6 = *(_DWORD *)(v4 + 24);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
      return sub_D46840(v9[1], a2, v2, v3);
    v12 = 1;
    while ( v10 != -4096 )
    {
      v13 = v12 + 1;
      v8 = v7 & (v12 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        return sub_D46840(v9[1], a2, v2, v3);
      v12 = v13;
    }
  }
  return sub_D46840(0, a2, v2, v3);
}
