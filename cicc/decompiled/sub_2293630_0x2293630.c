// Function: sub_2293630
// Address: 0x2293630
//
unsigned __int64 __fastcall sub_2293630(unsigned int *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned int v6; // edx
  unsigned __int64 result; // rax
  int v8; // edx
  unsigned __int64 v9; // r12

  v3 = a1[2];
  v4 = *(_QWORD *)a1;
  v5 = v3 + 1;
  v6 = a1[2];
  if ( v3 + 1 > (unsigned __int64)a1[3] )
  {
    if ( v4 > a2 || a2 >= v4 + 16 * v3 )
    {
      sub_AE4800(a1, v5);
      v3 = a1[2];
      v4 = *(_QWORD *)a1;
      v6 = a1[2];
    }
    else
    {
      v9 = a2 - v4;
      sub_AE4800(a1, v5);
      v4 = *(_QWORD *)a1;
      v3 = a1[2];
      a2 = *(_QWORD *)a1 + v9;
      v6 = a1[2];
    }
  }
  result = v4 + 16 * v3;
  if ( result )
  {
    v8 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a2 + 8) = 0;
    *(_DWORD *)(result + 8) = v8;
    *(_QWORD *)result = *(_QWORD *)a2;
    v6 = a1[2];
  }
  a1[2] = v6 + 1;
  return result;
}
