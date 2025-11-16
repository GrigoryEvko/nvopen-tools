// Function: sub_1BFBB30
// Address: 0x1bfbb30
//
__int64 __fastcall sub_1BFBB30(_DWORD *a1, __int64 a2, unsigned int a3, unsigned int *a4)
{
  unsigned int v4; // r8d
  unsigned int v8; // ecx
  unsigned int v9; // esi
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // edx
  unsigned int v13; // eax
  int v14; // [rsp-30h] [rbp-30h] BYREF
  char v15; // [rsp-2Ch] [rbp-2Ch]

  if ( !a1[3] )
    return 0;
  v4 = a3;
  if ( a1[7] < a3 )
  {
    sub_1C2ECF0(&v14, a2);
    v4 = 0;
    if ( v15 )
    {
      v8 = a1[11];
      v4 = a1[6];
      v9 = (v8 + v14 - 1) / v8;
      v10 = a1[3];
      if ( v4 < a3 )
      {
        *a4 = v10 / (v9 * v4 * v8);
      }
      else
      {
        v11 = v10 / (v9 * a3 * v8);
        v12 = a1[8];
        if ( v12 <= v11 )
        {
          *a4 = v12;
          return a3;
        }
        else
        {
          *a4 = v11 + 1;
          v4 = a1[7];
          v13 = a1[3] / ((v11 + 1) * v9 * a1[11]);
          if ( v13 >= v4 )
            return v13;
        }
      }
    }
  }
  return v4;
}
