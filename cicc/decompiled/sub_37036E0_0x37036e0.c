// Function: sub_37036E0
// Address: 0x37036e0
//
__int64 __fastcall sub_37036E0(__int64 a1, __int16 *a2)
{
  __int64 v2; // r14
  __int16 v4; // bx
  __int64 v5; // rax
  __int64 v6; // rdi
  int v7; // r8d
  __int16 v8; // ax
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // [rsp+0h] [rbp-70h]
  __int16 v15; // [rsp+16h] [rbp-5Ah] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h] BYREF
  __int16 v17; // [rsp+20h] [rbp-50h] BYREF
  __int64 v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h]

  v2 = a1 + 80;
  v4 = *a2;
  v5 = *(_QWORD *)(a1 + 136);
  v18 = 0;
  v6 = *(_QWORD *)(a1 + 104);
  v19 = 0;
  v17 = v4;
  v14 = v5;
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 16LL))(v6);
  v8 = __ROL2__(v4, 8);
  if ( v7 != 1 )
    v4 = v8;
  v15 = v4;
  sub_3719260(&v16, v2, &v15, 2);
  if ( (v16 & 0xFFFFFFFFFFFFFFFELL) != 0
    || (sub_370EDF0(&v16, a1 + 144), (v16 & 0xFFFFFFFFFFFFFFFELL) != 0)
    || (sub_37141A0(&v16, a1 + 144, &v17, a2), (v16 & 0xFFFFFFFFFFFFFFFELL) != 0)
    || (sub_370D250(&v16, a1 + 144, &v17), (v16 & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    BUG();
  }
  sub_3702B60(v2);
  result = sub_3702F70(a1);
  if ( (unsigned int)result > 0xFEF8 )
    return sub_3702F90(a1, (char *)v14, v10, v11, v12, v13);
  return result;
}
