// Function: sub_3170240
// Address: 0x3170240
//
__int64 __fastcall sub_3170240(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  unsigned int v7; // r15d
  unsigned int v9; // edx
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  int v12; // ebx
  unsigned int v13; // r14d
  __int64 v14; // rax
  unsigned int v15; // eax

  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_9;
  v6 = *(__int64 **)(a2 + 8);
  a4 = *(unsigned int *)(a2 + 20);
  a3 = &v6[a4];
  if ( v6 == a3 )
  {
LABEL_8:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = a4 + 1;
      *a3 = a1;
      ++*(_QWORD *)a2;
LABEL_10:
      v7 = sub_24F3200(a1);
      if ( !(_BYTE)v7 )
      {
        v10 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v10 != a1 + 48 )
        {
          if ( !v10 )
            BUG();
          v11 = v10 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 <= 0xA )
          {
            v12 = sub_B46E30(v11);
            if ( v12 )
            {
              v13 = 0;
              while ( 1 )
              {
                v14 = sub_B46EC0(v11, v13);
                v15 = sub_3170240(v14, a2);
                if ( (_BYTE)v15 )
                  break;
                if ( v12 == ++v13 )
                  return v7;
              }
              return v15;
            }
          }
        }
      }
      return v7;
    }
LABEL_9:
    sub_C8CC70(a2, a1, (__int64)a3, a4, a5, a6);
    v7 = v9;
    if ( !(_BYTE)v9 )
      return v7;
    goto LABEL_10;
  }
  while ( a1 != *v6 )
  {
    if ( a3 == ++v6 )
      goto LABEL_8;
  }
  return 0;
}
