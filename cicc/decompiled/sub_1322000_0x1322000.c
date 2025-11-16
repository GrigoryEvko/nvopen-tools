// Function: sub_1322000
// Address: 0x1322000
//
__int64 __fastcall sub_1322000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int64 *a5,
        __int64 *a6,
        __int64 a7)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 result; // rax
  unsigned __int64 v13; // rdi
  char *v14; // rbx
  char *v15; // rcx
  unsigned int v16; // ebx
  unsigned int v17; // ebx
  unsigned int v18; // eax
  __int64 v19; // rsi
  _QWORD v20[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a4 && a5 )
  {
    v10 = sub_1317030(a1, a2);
    v11 = *a5;
    v20[0] = v10;
    if ( v11 != 8 )
    {
      if ( v11 > 8 )
        v11 = 8;
      if ( (unsigned int)v11 >= 8 )
      {
        *a4 = v10;
        v13 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *(__int64 *)((char *)a4 + (unsigned int)v11 - 8) = *(_QWORD *)((char *)&v20[-1] + (unsigned int)v11);
        v14 = (char *)a4 - v13;
        v15 = (char *)((char *)v20 - v14);
        v16 = (v11 + (_DWORD)v14) & 0xFFFFFFF8;
        if ( v16 >= 8 )
        {
          v17 = v16 & 0xFFFFFFF8;
          v18 = 0;
          do
          {
            v19 = v18;
            v18 += 8;
            *(_QWORD *)(v13 + v19) = *(_QWORD *)&v15[v19];
          }
          while ( v18 < v17 );
        }
      }
      else if ( (v11 & 4) != 0 )
      {
        *(_DWORD *)a4 = v20[0];
        *(_DWORD *)((char *)a4 + (unsigned int)v11 - 4) = *(_DWORD *)((char *)v20 + (unsigned int)v11 - 4);
      }
      else if ( (_DWORD)v11 )
      {
        *(_BYTE *)a4 = v20[0];
        if ( (v11 & 2) != 0 )
          *(_WORD *)((char *)a4 + (unsigned int)v11 - 2) = *(_WORD *)((char *)v20 + (unsigned int)v11 - 2);
      }
      *a5 = v11;
      return 22;
    }
    *a4 = v10;
  }
  if ( a6 )
  {
    result = 22;
    if ( a7 != 8 )
      return result;
    if ( (unsigned __int8)sub_1317040(*a6) )
      return 14;
  }
  return 0;
}
