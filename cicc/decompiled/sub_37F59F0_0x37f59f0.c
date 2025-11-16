// Function: sub_37F59F0
// Address: 0x37f59f0
//
__int64 __fastcall sub_37F59F0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v4; // eax
  __int64 v5; // rcx
  int v6; // r9d
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r10
  int v10; // ebx
  int v12; // eax
  int v13; // r11d

  v4 = *(_DWORD *)(a1 + 488);
  v5 = *(_QWORD *)(a1 + 472);
  if ( !v4 )
  {
LABEL_7:
    v10 = 0;
    return v10 - (unsigned int)sub_37F56A0(a1, a2, a3);
  }
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v12 = 1;
    while ( v9 != -4096 )
    {
      v13 = v12 + 1;
      v7 = v6 & (v12 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v12 = v13;
    }
    goto LABEL_7;
  }
LABEL_3:
  v10 = *((_DWORD *)v8 + 2);
  return v10 - (unsigned int)sub_37F56A0(a1, a2, a3);
}
