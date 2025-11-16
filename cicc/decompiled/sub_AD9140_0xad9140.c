// Function: sub_AD9140
// Address: 0xad9140
//
unsigned __int8 *__fastcall sub_AD9140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int8 *v10; // r13
  int v11; // edx
  _QWORD *i; // rbx
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+18h] [rbp-58h]
  __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v17; // [rsp+28h] [rbp-48h]

  v4 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v14 = sub_BCAC60(a1);
  v8 = sub_C33340(a1, a2, v5, v6, v7);
  v9 = v8;
  if ( v14 == v8 )
    sub_C3C500(&v16, v8, 0);
  else
    sub_C373C0(&v16, v14, 0);
  if ( v16 == v9 )
    sub_C3D480(&v16, 0, (unsigned __int8)a2, a3);
  else
    sub_C36070(&v16, 0, (unsigned __int8)a2, a3);
  v10 = (unsigned __int8 *)sub_AC8EA0(*(__int64 **)v4, &v16);
  v11 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    BYTE4(v15) = (_BYTE)v11 == 18;
    LODWORD(v15) = *(_DWORD *)(v4 + 32);
    v10 = (unsigned __int8 *)sub_AD5E10(v15, v10);
  }
  if ( v9 == v16 )
  {
    if ( v17 )
    {
      for ( i = &v17[3 * *(v17 - 1)]; v17 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v16);
  }
  return v10;
}
