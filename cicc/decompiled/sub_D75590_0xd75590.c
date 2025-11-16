// Function: sub_D75590
// Address: 0xd75590
//
__int64 *__fastcall sub_D75590(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  int v7; // ecx
  unsigned __int64 v8; // rax
  int v9; // ecx
  int v10; // r10d
  __int64 *v11; // rcx
  __int64 v12; // r11
  int v14; // ecx
  int v15; // ebx
  __int64 v16; // r11

  if ( (_DWORD)a4 == 2 )
  {
    a5 = *a1;
    v6 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v6 == a3 + 48 )
    {
      v8 = 0;
    }
    else
    {
      if ( !v6 )
        BUG();
      v7 = *(unsigned __int8 *)(v6 - 24);
      v8 = v6 - 24;
      if ( (unsigned int)(v7 - 30) >= 0xB )
        v8 = 0;
    }
    v9 = *(_DWORD *)(a5 + 56);
    a6 = *(_QWORD *)(a5 + 40);
    if ( v9 )
    {
      v10 = v9 - 1;
      a5 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v11 = (__int64 *)(a6 + 16 * a5);
      v12 = *v11;
      if ( v8 == *v11 )
      {
        a5 = v11[1];
        if ( a5 )
          return sub_D75380(a1, a2, v11[1], (__int64)v11, a5, a6);
      }
      else
      {
        v14 = 1;
        if ( v12 != -4096 )
        {
          while ( 1 )
          {
            v15 = v14 + 1;
            a5 = v10 & (unsigned int)(v14 + a5);
            v11 = (__int64 *)(a6 + 16LL * (unsigned int)a5);
            v16 = *v11;
            if ( v8 == *v11 )
              break;
            v14 = v15;
            if ( v16 == -4096 )
              goto LABEL_11;
          }
          a5 = v11[1];
          if ( !a5 )
          {
LABEL_11:
            a4 = 1;
            return sub_D753A0(a1, a2, a3, a4, a5, a6);
          }
          return sub_D75380(a1, a2, v11[1], (__int64)v11, a5, a6);
        }
      }
    }
    a4 = 1;
  }
  return sub_D753A0(a1, a2, a3, a4, a5, a6);
}
