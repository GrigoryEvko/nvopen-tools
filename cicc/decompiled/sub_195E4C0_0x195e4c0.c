// Function: sub_195E4C0
// Address: 0x195e4c0
//
void __fastcall sub_195E4C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  int v6; // r9d
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 v11; // rdi
  int v12; // eax
  int v13; // r11d

  v4 = *(_DWORD *)(a1 + 184);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = *(_QWORD *)(a1 + 168);
    v8 = (v4 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a4 == *v9 )
    {
LABEL_3:
      v11 = v9[1];
      if ( v11 )
        sub_135BB20(v11, a2, a3);
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v13 = v12 + 1;
        v8 = v6 & (v12 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( a4 == *v9 )
          goto LABEL_3;
        v12 = v13;
      }
    }
  }
}
