// Function: sub_2CAF330
// Address: 0x2caf330
//
__int64 __fastcall sub_2CAF330(__int64 a1, int a2, __int64 a3, __int64 *a4, __int64 a5)
{
  unsigned int v8; // edx
  unsigned int v9; // eax
  __int64 v10; // r12
  _QWORD *v11; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax

  v8 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) >> 8;
  v9 = *(_DWORD *)(a3 + 8) >> 8;
  if ( v8 > v9 )
  {
    v13 = sub_BD2C40(72, 1u);
    v10 = (__int64)v13;
    if ( v13 )
      sub_B51510((__int64)v13, a1, a3, a5, 0, 0);
    goto LABEL_6;
  }
  v10 = a1;
  if ( v8 < v9 )
  {
    if ( a2 == 2 )
    {
      v14 = sub_BD2C40(72, 1u);
      v10 = (__int64)v14;
      if ( v14 )
        sub_B515B0((__int64)v14, a1, a3, a5, 0, 0);
    }
    else
    {
      v11 = sub_BD2C40(72, 1u);
      v10 = (__int64)v11;
      if ( v11 )
        sub_B51650((__int64)v11, a1, a3, a5, 0, 0);
    }
LABEL_6:
    sub_B43DD0(v10, *a4);
    *a4 = v10;
  }
  return v10;
}
