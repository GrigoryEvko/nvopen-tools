// Function: sub_269E480
// Address: 0x269e480
//
__int64 __fastcall sub_269E480(__int64 *a1, unsigned __int8 *a2)
{
  int v2; // r12d
  __int64 v4; // rdx
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r12
  unsigned __int64 v11; // rax
  unsigned __int8 (__fastcall *v12)(__int64, __int64 (__fastcall *)(__int64, unsigned __int8 *), unsigned __int8 **, __int64); // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int8 *v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 *v17; // [rsp+30h] [rbp-40h]

  v2 = *a2;
  if ( (unsigned __int8)(v2 - 34) > 0x33u )
  {
    if ( !(unsigned __int8)sub_B46490((__int64)a2) )
      return 1;
LABEL_9:
    v13 = a1[1];
    v15 = a2;
    sub_269CCD0(v13 + 248, (__int64 *)&v15);
    return 1;
  }
  v4 = 0x8000000000041LL;
  if ( !_bittest64(&v4, (unsigned int)(v2 - 34)) )
  {
    if ( (unsigned __int8)sub_B46490((__int64)a2) )
    {
      if ( (_BYTE)v2 != 62 )
        goto LABEL_9;
      v5 = *a1;
      v6 = sub_250D2C0(*((_QWORD *)a2 - 4), 0);
      v8 = sub_252AE70(v5, v6, v7, a1[1], 1, 0, 1);
      v9 = *a1;
      v10 = v8;
      v11 = sub_B43CB0((__int64)a2);
      sub_250D230((unsigned __int64 *)&v15, v11, 4, 0);
      v14 = sub_269DF00(v9, (__int64)v15, v16, a1[1], 1, 0, 1);
      if ( !v10 )
        goto LABEL_9;
      v12 = *(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, unsigned __int8 *), unsigned __int8 **, __int64))(*(_QWORD *)v10 + 112LL);
      v15 = (unsigned __int8 *)*a1;
      v16 = a1[1];
      v17 = &v14;
      if ( !v12(v10, sub_266F4F0, &v15, 2) )
        goto LABEL_9;
    }
  }
  return 1;
}
