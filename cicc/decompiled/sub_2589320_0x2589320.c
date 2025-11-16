// Function: sub_2589320
// Address: 0x2589320
//
_BOOL8 __fastcall sub_2589320(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  char v3; // r8
  _BOOL8 result; // rax
  __int16 v5; // ax
  __int64 (*v6)(void); // rdx
  __int16 v7; // dx
  __int16 v8; // ax
  bool v9; // [rsp+7h] [rbp-39h] BYREF
  _WORD *v10; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 v11[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = sub_250C680((__int64 *)(a1 + 72));
  if ( !v2 )
    goto LABEL_5;
  sub_250D230(v11, v2, 6, 0);
  v10 = 0;
  v3 = sub_25890A0(a2, a1, (__int64 *)v11, 0, &v9, 0, (__int64 *)&v10);
  result = 1;
  if ( v3 )
    return result;
  if ( v10 && (v5 = v10[49], (v5 & 3) == 3) )
  {
    v6 = *(__int64 (**)(void))(*(_QWORD *)v10 + 48LL);
    if ( (char *)v6 != (char *)sub_2534F70 )
      v5 = *(_WORD *)(v6() + 10);
    v7 = *(_WORD *)(a1 + 98);
    v8 = *(_WORD *)(a1 + 96) | v7 & v5;
    *(_WORD *)(a1 + 98) = v8;
    return v7 == v8;
  }
  else
  {
LABEL_5:
    *(_WORD *)(a1 + 98) = *(_WORD *)(a1 + 96);
    return 0;
  }
}
