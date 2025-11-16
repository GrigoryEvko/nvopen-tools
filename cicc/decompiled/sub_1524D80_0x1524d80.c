// Function: sub_1524D80
// Address: 0x1524d80
//
void __fastcall sub_1524D80(_DWORD *a1, unsigned int a2, int a3)
{
  int v3; // ecx
  int v4; // r13d
  unsigned int v5; // ecx
  int v6; // r13d
  __int64 v7; // r15
  char v8; // r12
  __int64 v9; // rdx
  unsigned int v10; // eax
  int v11; // edx

  v3 = a1[2];
  v4 = a2 << v3;
  v5 = a3 + v3;
  v6 = a1[3] | v4;
  a1[3] = v6;
  if ( v5 > 0x1F )
  {
    v7 = *(_QWORD *)a1;
    v8 = a3;
    v9 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v9 <= 3 )
    {
      sub_16CD150(*(_QWORD *)a1, v7 + 16, v9 + 4, 1);
      v9 = *(unsigned int *)(v7 + 8);
    }
    *(_DWORD *)(*(_QWORD *)v7 + v9) = v6;
    v10 = 0;
    *(_DWORD *)(v7 + 8) += 4;
    v11 = a1[2];
    if ( v11 )
      v10 = a2 >> (32 - v11);
    a1[3] = v10;
    a1[2] = (v8 + (_BYTE)v11) & 0x1F;
  }
  else
  {
    a1[2] = v5;
  }
}
