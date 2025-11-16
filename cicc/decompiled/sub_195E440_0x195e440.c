// Function: sub_195E440
// Address: 0x195e440
//
void __fastcall sub_195E440(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v4; // r8
  int v5; // edi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdi
  int v10; // eax
  int v11; // r10d

  v3 = *(_DWORD *)(a1 + 184);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a1 + 168);
    v5 = v3 - 1;
    v6 = (v3 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a3 == *v7 )
    {
LABEL_3:
      v9 = v7[1];
      if ( v9 )
        sub_1359860(v9, a2);
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v11 = v10 + 1;
        v6 = v5 & (v10 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a3 == *v7 )
          goto LABEL_3;
        v10 = v11;
      }
    }
  }
}
