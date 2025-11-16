// Function: sub_AD9500
// Address: 0xad9500
//
__int64 __fastcall sub_AD9500(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int8 *v9; // r13
  int v10; // edx
  int v11; // eax
  _QWORD *i; // rbx
  __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+8h] [rbp-48h]

  v2 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v3 = sub_BCAC60(a1);
  v7 = sub_C33340(a1, a2, v4, v5, v6);
  v8 = v7;
  if ( v3 == v7 )
    sub_C3C500(&v14, v7, 0);
  else
    sub_C373C0(&v14, v3, 0);
  if ( v14 == v8 )
    sub_C3CF20(&v14, (unsigned __int8)a2);
  else
    sub_C36EF0(&v14, (unsigned __int8)a2);
  v9 = (unsigned __int8 *)sub_AC8EA0(*(__int64 **)v2, &v14);
  if ( v14 == v8 )
  {
    if ( v15 )
    {
      for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v14);
  }
  v10 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned int)(v10 - 17) > 1 )
    return (__int64)v9;
  v11 = *(_DWORD *)(v2 + 32);
  BYTE4(v14) = (_BYTE)v10 == 18;
  LODWORD(v14) = v11;
  return sub_AD5E10(v14, v9);
}
