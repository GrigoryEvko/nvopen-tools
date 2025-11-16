// Function: sub_39C95F0
// Address: 0x39c95f0
//
void __fastcall sub_39C95F0(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int8 *v4; // r14
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // edx
  unsigned __int8 **v10; // rax
  unsigned __int8 *v11; // rdi
  __int64 v12; // rcx
  int v13; // eax
  int v14; // r8d

  v4 = sub_39A23D0((__int64)a1, a2);
  if ( !sub_39C7370((__int64)a1) || (unsigned __int8)sub_3989C80(a1[25]) )
    v5 = a1[26] + 296LL;
  else
    v5 = (__int64)(a1 + 108);
  v6 = *(_DWORD *)(v5 + 24);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (unsigned __int8 **)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_5:
      v12 = (__int64)v10[1];
      if ( v12 )
      {
        if ( v4 )
          sub_39A3B20((__int64)a1, (__int64)v4, 49, v12);
        return;
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != (unsigned __int8 *)-8LL )
      {
        v14 = v13 + 1;
        v9 = v8 & (v13 + v9);
        v10 = (unsigned __int8 **)(v7 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_5;
        v13 = v14;
      }
    }
  }
  if ( v4 )
    sub_39C9540(a1, (__int64)a2, (__int64)v4);
}
