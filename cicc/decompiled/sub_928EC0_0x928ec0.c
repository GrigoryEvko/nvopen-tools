// Function: sub_928EC0
// Address: 0x928ec0
//
__int64 __fastcall sub_928EC0(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  char v5; // al
  unsigned int **v6; // r15
  bool v7; // zf
  unsigned int *v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v10; // r12
  unsigned int *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rsi
  _QWORD v20[4]; // [rsp+0h] [rbp-90h] BYREF
  char v21; // [rsp+20h] [rbp-70h]
  char v22; // [rsp+21h] [rbp-6Fh]
  _BYTE v23[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v24; // [rsp+50h] [rbp-40h]

  v5 = sub_91B6F0(a4);
  v6 = *(unsigned int ***)(a1 + 8);
  v22 = 1;
  v7 = v5 == 0;
  v21 = 3;
  v20[0] = "rem";
  v8 = v6[10];
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v8 + 16LL);
  if ( !v7 )
  {
    if ( v9 == sub_9202E0 )
    {
      if ( *a2 > 0x15u || *a3 > 0x15u )
      {
LABEL_18:
        v24 = 257;
        v10 = sub_B504D0(23, a2, a3, v23, 0, 0);
        (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v6[11]
                                                                                                  + 16LL))(
          v6[11],
          v10,
          v20,
          v6[7],
          v6[8]);
        v16 = *v6;
        v17 = (__int64)&(*v6)[4 * *((unsigned int *)v6 + 2)];
        if ( *v6 != (unsigned int *)v17 )
        {
          do
          {
            v18 = *((_QWORD *)v16 + 1);
            v19 = *v16;
            v16 += 4;
            sub_B99FD0(v10, v19, v18);
          }
          while ( (unsigned int *)v17 != v16 );
        }
        return v10;
      }
      if ( (unsigned __int8)sub_AC47B0(23) )
        v10 = sub_AD5570(23, a2, a3, 0, 0);
      else
        v10 = sub_AABE40(23, a2, a3);
    }
    else
    {
      v10 = v9((__int64)v8, 23u, a2, a3);
    }
    if ( v10 )
      return v10;
    goto LABEL_18;
  }
  if ( v9 != sub_9202E0 )
  {
    v10 = v9((__int64)v8, 22u, a2, a3);
    goto LABEL_14;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(22) )
      v10 = sub_AD5570(22, a2, a3, 0, 0);
    else
      v10 = sub_AABE40(22, a2, a3);
LABEL_14:
    if ( v10 )
      return v10;
  }
  v24 = 257;
  v10 = sub_B504D0(22, a2, a3, v23, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v6[11] + 16LL))(
    v6[11],
    v10,
    v20,
    v6[7],
    v6[8]);
  v12 = *v6;
  v13 = (__int64)&(*v6)[4 * *((unsigned int *)v6 + 2)];
  if ( *v6 != (unsigned int *)v13 )
  {
    do
    {
      v14 = *((_QWORD *)v12 + 1);
      v15 = *v12;
      v12 += 4;
      sub_B99FD0(v10, v15, v14);
    }
    while ( (unsigned int *)v13 != v12 );
  }
  return v10;
}
