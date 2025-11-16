// Function: sub_289A360
// Address: 0x289a360
//
__int64 __fastcall sub_289A360(_QWORD *a1, _BYTE *a2, __int64 a3, int a4, _BYTE *a5)
{
  __int64 v6; // rcx
  int v7; // eax
  unsigned int **v8; // rdi
  __int64 v9; // r12
  unsigned int v11; // [rsp+1Ch] [rbp-54h]
  _QWORD v14[4]; // [rsp+30h] [rbp-40h] BYREF
  char v15; // [rsp+50h] [rbp-20h]
  char v16; // [rsp+51h] [rbp-1Fh]

  v6 = *(_QWORD *)(*a1 + 8LL);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
    LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  v8 = (unsigned int **)a1[1];
  if ( (unsigned __int8)v7 <= 3u || (_BYTE)v7 == 5 || (v7 & 0xFD) == 4 )
  {
    v16 = 1;
    v14[0] = "mmul";
    v15 = 3;
    v9 = sub_A826E0(v8, a2, a5, v11, (__int64)v14, 0);
  }
  else
  {
    v16 = 1;
    v14[0] = "mmul";
    v15 = 3;
    v9 = sub_A81850(v8, a2, a5, (__int64)v14, 0, 0);
  }
  sub_2896BA0(a1[2], v9, a3, a4);
  return v9;
}
