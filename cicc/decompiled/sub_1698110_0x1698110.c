// Function: sub_1698110
// Address: 0x1698110
//
void __fastcall sub_1698110(int *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ebx
  char **v4; // rax
  int v5; // edx
  __int64 v6; // r13
  unsigned __int64 v7; // r14
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rbx

  v3 = a1[1];
  if ( v3 )
  {
    v4 = (char **)sub_16946D0(a2, *a1);
    sub_1695BA0(v4, v3);
    v5 = a1[1];
    v6 = (__int64)a1 + ((v5 + 15) & 0xFFFFFFF8);
    if ( v5 )
    {
      v7 = 0;
      do
      {
        v8 = *((unsigned __int8 *)a1 + v7 + 8);
        v9 = (unsigned int)v7++;
        v10 = v8;
        sub_1697FB0(a2, *a1, v9, v6, v8, a3);
        v6 += 16 * v10;
      }
      while ( (unsigned int)a1[1] > v7 );
    }
  }
}
