// Function: sub_310F9A0
// Address: 0x310f9a0
//
__int64 __fastcall sub_310F9A0(__int64 a1, __int64 **a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  int v7; // edx
  __int64 result; // rax
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  _QWORD v15[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_BA91D0((__int64)a2, "cfguard", 7u);
  if ( !v4 || (v5 = *(_QWORD *)(v4 + 136)) == 0 )
  {
    result = 0;
    if ( *(_DWORD *)a1 != 2 )
      return result;
LABEL_8:
    v9 = sub_BCE3C0(*a2, 0);
    v10 = *a2;
    v15[0] = v9;
    v11 = (__int64 *)sub_BCB120(v10);
    *(_QWORD *)(a1 + 32) = sub_BCF480(v11, v15, 1, 0);
    v12 = sub_BCE3C0(*a2, 0);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 40) = v12;
    v15[0] = a2;
    v15[1] = a1;
    *(_QWORD *)(a1 + 48) = sub_BA8D20(
                             (__int64)a2,
                             v13,
                             v14,
                             v12,
                             (__int64 (__fastcall *)(__int64))sub_310FA90,
                             (__int64)v15);
    return 1;
  }
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v7 = (int)v6;
  *(_DWORD *)a1 = (_DWORD)v6;
  result = 0;
  if ( v7 == 2 )
    goto LABEL_8;
  return result;
}
