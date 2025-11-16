// Function: sub_AD8DD0
// Address: 0xad8dd0
//
unsigned __int8 *__fastcall sub_AD8DD0(__int64 a1, double a2)
{
  __int64 v2; // rbx
  __int64 *v3; // r12
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int8 *v10; // r12
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  _QWORD *i; // rbx
  char v16; // [rsp+1Fh] [rbp-71h] BYREF
  __int64 v17; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v18; // [rsp+28h] [rbp-68h]
  __int64 v19[10]; // [rsp+40h] [rbp-50h] BYREF

  v2 = a1;
  v3 = *(__int64 **)a1;
  v4 = sub_C33320(a1);
  sub_C3B1B0(v19, a2);
  sub_C407B0(&v17, v19, v4);
  sub_C338F0(v19);
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v5 = sub_BCAC60(a1);
  sub_C41640(&v17, v5, 1, &v16);
  v6 = (__int64)v3;
  v7 = &v17;
  v10 = (unsigned __int8 *)sub_AC8EA0(v3, &v17);
  v11 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    v12 = *(_DWORD *)(v2 + 32);
    v7 = (__int64 *)v10;
    BYTE4(v19[0]) = (_BYTE)v11 == 18;
    LODWORD(v19[0]) = v12;
    v6 = v19[0];
    v10 = (unsigned __int8 *)sub_AD5E10(v19[0], v10);
  }
  v13 = sub_C33340(v6, v7, v11, v8, v9);
  if ( v17 == v13 )
  {
    if ( v18 )
    {
      for ( i = &v18[3 * *(v18 - 1)]; v18 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v17);
  }
  return v10;
}
