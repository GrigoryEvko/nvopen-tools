// Function: sub_1009690
// Address: 0x1009690
//
__int64 __fastcall sub_1009690(double *a1, __int64 a2)
{
  _DWORD *v2; // r15
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int v6; // r13d
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 v10; // r12
  _DWORD *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *i; // r12
  double v16; // [rsp+8h] [rbp-78h]
  double v17; // [rsp+8h] [rbp-78h]
  void *v18; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-68h]
  __int64 v20[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( *(_BYTE *)a2 == 18 )
  {
    v16 = *a1;
    v2 = sub_C33320();
    sub_C3B1B0((__int64)v20, v16);
    sub_C407B0(&v18, v20, v2);
    sub_C338F0((__int64)v20);
    sub_C41640((__int64 *)&v18, *(_DWORD **)(a2 + 24), 1, (bool *)v20);
    v6 = sub_AC3090(a2, &v18, v3, v4, v5);
    if ( v18 == sub_C33340() )
    {
      if ( v19 )
      {
        for ( i = &v19[3 * *(v19 - 1)]; v19 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0(i - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v18);
    }
  }
  else
  {
    v8 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v8 <= 1
      && *(_BYTE *)a2 <= 0x15u
      && (v9 = sub_AD7630(a2, 0, v8), (v10 = (__int64)v9) != 0)
      && *v9 == 18 )
    {
      v17 = *a1;
      v11 = sub_C33320();
      sub_C3B1B0((__int64)v20, v17);
      sub_C407B0(&v18, v20, v11);
      sub_C338F0((__int64)v20);
      sub_C41640((__int64 *)&v18, *(_DWORD **)(v10 + 24), 1, (bool *)v20);
      v6 = sub_AC3090(v10, &v18, v12, v13, v14);
      sub_91D830(&v18);
    }
    else
    {
      return 0;
    }
  }
  return v6;
}
