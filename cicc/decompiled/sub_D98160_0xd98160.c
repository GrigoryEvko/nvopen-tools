// Function: sub_D98160
// Address: 0xd98160
//
__int64 __fastcall sub_D98160(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  int v12; // eax
  int v13; // r8d

  v3 = sub_AA54C0(a2);
  if ( !v3 )
  {
    v5 = *(_QWORD *)(a1 + 48);
    v6 = *(_QWORD *)(v5 + 8);
    v7 = *(_DWORD *)(v5 + 24);
    if ( v7 )
    {
      v8 = v7 - 1;
      v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
      {
LABEL_5:
        v3 = v10[1];
        if ( v3 )
          return sub_D47840(v10[1]);
      }
      else
      {
        v12 = 1;
        while ( v11 != -4096 )
        {
          v13 = v12 + 1;
          v9 = v8 & (v12 + v9);
          v10 = (__int64 *)(v6 + 16LL * v9);
          v11 = *v10;
          if ( a2 == *v10 )
            goto LABEL_5;
          v12 = v13;
        }
      }
    }
  }
  return v3;
}
