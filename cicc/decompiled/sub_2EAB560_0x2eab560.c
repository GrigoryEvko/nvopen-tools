// Function: sub_2EAB560
// Address: 0x2eab560
//
unsigned __int64 __fastcall sub_2EAB560(
        char *a1,
        int a2,
        unsigned __int8 a3,
        unsigned __int8 a4,
        char a5,
        char a6,
        unsigned __int8 a7,
        unsigned __int8 a8)
{
  int v8; // eax
  __int64 v11; // rdx
  unsigned __int8 v12; // r11
  unsigned __int8 v13; // r10
  char v14; // r15
  __int64 v15; // r14
  unsigned __int64 result; // rax
  char v17; // [rsp+14h] [rbp-3Ch]
  char v18; // [rsp+18h] [rbp-38h]

  v8 = a2;
  v11 = *((_QWORD *)a1 + 2);
  v12 = a7;
  v13 = a8;
  v14 = *a1;
  if ( v11 )
  {
    v15 = *(_QWORD *)(v11 + 24);
    if ( v15 )
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( v15 )
      {
        v15 = *(_QWORD *)(v15 + 32);
        if ( v15 )
        {
          if ( !v14 )
          {
            v17 = a6;
            v18 = a5;
            sub_2EBEB60(v15, a1);
            v11 = *((_QWORD *)a1 + 2);
            v13 = a8;
            v12 = a7;
            a6 = v17;
            a5 = v18;
            v8 = a2;
          }
        }
      }
    }
    if ( ((v11 != 0) & (a3 ^ 1)) != 0 && (unsigned __int16)(*(_WORD *)(v11 + 68) - 14) <= 4u )
      v13 = (v11 != 0) & (a3 ^ 1);
  }
  else
  {
    v15 = 0;
  }
  *((_DWORD *)a1 + 2) = v8;
  *((_QWORD *)a1 + 3) = 0;
  result = *(_QWORD *)a1 & 0xFFFFFFF00FF00000LL
         | (((unsigned __int64)v13 << 35)
          | ((unsigned __int64)v12 << 32)
          | ((unsigned __int64)(unsigned __int8)(a5 | a6) << 30)
          | ((unsigned __int64)a3 << 28)
          | ((unsigned __int64)a4 << 29))
         & 0xFF00FFFFFLL;
  *(_QWORD *)a1 = result;
  if ( v14 )
    *((_WORD *)a1 + 1) &= 0xF00Fu;
  if ( v15 )
    return sub_2EBEAE0(v15, a1);
  return result;
}
