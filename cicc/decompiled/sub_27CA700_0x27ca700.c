// Function: sub_27CA700
// Address: 0x27ca700
//
__int64 __fastcall sub_27CA700(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r15
  char v7; // al
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // r12
  _QWORD *v12; // r14
  _QWORD *v13; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+0h] [rbp-40h]
  void *v16; // [rsp+8h] [rbp-38h]

  if ( a2 == 13 )
  {
    v16 = sub_DC7ED0;
  }
  else
  {
    if ( a2 != 15 )
      BUG();
    v16 = sub_DCC810;
  }
  v6 = (__int64 *)*a1;
  v14 = *(_QWORD *)a1[2];
  v7 = sub_B532B0(*(_DWORD *)a1[1]);
  if ( (unsigned __int8)sub_DDCBC0(v6, a2, v7, a3, a4, v14) )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, _QWORD))v16)(*a1, a3, a4, 0, 0);
  v9 = sub_D95540(a3);
  v10 = *(_DWORD *)(v9 + 8) >> 8;
  if ( v10 > (unsigned int)qword_4FFCE28 )
    return 0;
  v11 = sub_BCCE00(*(_QWORD **)v9, 2 * v10);
  v15 = *a1;
  v12 = sub_DC5000(*a1, a4, v11, 0);
  v13 = sub_DC5000(*a1, a3, v11, 0);
  return ((__int64 (__fastcall *)(__int64, _QWORD *, _QWORD *, _QWORD, _QWORD))v16)(v15, v13, v12, 0, 0);
}
