// Function: sub_1EBB620
// Address: 0x1ebb620
//
__int64 __fastcall sub_1EBB620(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r11
  __int64 v5; // r10
  __int64 i; // r8
  int *v9; // rdx
  _DWORD *v10; // rax
  int *v11; // rax
  int v12; // ecx
  _DWORD *v14; // rdx
  _DWORD *v15; // rcx

  v4 = a2;
  v5 = (a3 - 1) / 2;
  if ( a2 < v5 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1) - 1;
      v11 = (int *)(a1 + 16 * (i + 1));
      v9 = (int *)(a1 + 8 * a2);
      v12 = *v9;
      if ( *v11 >= (unsigned int)*v9 )
      {
        if ( *v11 == v12 )
        {
          if ( v11[1] >= (unsigned int)v9[1] )
          {
            v9 = (int *)(a1 + 16 * (i + 1));
            a2 = 2 * (i + 1);
          }
        }
        else
        {
          v12 = *v11;
          v9 = (int *)(a1 + 16 * (i + 1));
          a2 = 2 * (i + 1);
        }
      }
      v10 = (_DWORD *)(a1 + 8 * i);
      *v10 = v12;
      v10[1] = v9[1];
      if ( a2 >= v5 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == a2 )
  {
    v14 = (_DWORD *)(a1 + 8 * a2);
    v15 = (_DWORD *)(a1 + 8 * (2 * a2 + 1));
    *v14 = *v15;
    a2 = 2 * a2 + 1;
    v14[1] = v15[1];
  }
  return sub_1EBB580(a1, a2, v4, a4);
}
