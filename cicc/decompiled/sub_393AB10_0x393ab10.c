// Function: sub_393AB10
// Address: 0x393ab10
//
__int64 *__fastcall sub_393AB10(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // eax
  int v9; // [rsp+4h] [rbp-4Ch] BYREF
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  int *v12; // [rsp+18h] [rbp-38h] BYREF
  __int64 v13; // [rsp+20h] [rbp-30h] BYREF
  __int64 v14[5]; // [rsp+28h] [rbp-28h] BYREF

  v4 = &v13;
  v5 = *a3;
  *a3 = 0;
  v12 = &v9;
  v9 = 0;
  v13 = v5 | 1;
  v10 = 0;
  v11 = 0;
  sub_393A930(v14, &v13, &v12);
  if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v14[0] = v14[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_16BCAE0(v14, (__int64)&v13, v6);
  }
  if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v13, (__int64)&v13, v6);
  if ( (v11 & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v11, (__int64)&v13, v6);
  v7 = v9;
  LODWORD(v14[0]) = v9;
  *(_DWORD *)(a2 + 8) = v9;
  if ( v7 )
  {
    v4 = v14;
    sub_3939040(a1, (int *)v14);
  }
  else
  {
    *a1 = 1;
  }
  if ( (v10 & 1) != 0 || (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v10, (__int64)v4, v6);
  return a1;
}
