// Function: sub_23BFF20
// Address: 0x23bff20
//
void __fastcall sub_23BFF20(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rsi
  __int64 *v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-B0h]
  __int64 v19; // [rsp+8h] [rbp-A8h]
  _BYTE *v20; // [rsp+20h] [rbp-90h] BYREF
  __int64 v21; // [rsp+28h] [rbp-88h]
  __int64 v22; // [rsp+30h] [rbp-80h]
  __int64 v23; // [rsp+38h] [rbp-78h]
  __int64 v24; // [rsp+40h] [rbp-70h]
  __int64 v25; // [rsp+48h] [rbp-68h]
  __int64 *v26; // [rsp+50h] [rbp-60h]
  __int64 v27; // [rsp+58h] [rbp-58h]
  _QWORD v28[10]; // [rsp+60h] [rbp-50h] BYREF

  if ( !byte_4FDE3D0 && (unsigned int)sub_2207590((__int64)&byte_4FDE3D0) )
  {
    dword_4FDE3F0 = -1;
    qword_4FDE3E0 = (__int64)&dword_4FDE3F0;
    qword_4FDE3E8 = 0xC00000001LL;
    __cxa_atexit((void (*)(void *))sub_BC4ED0, &qword_4FDE3E0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FDE3D0);
  }
  v6 = a2[1];
  v7 = *a2;
  v26 = v28;
  v28[1] = v6;
  v28[0] = v7;
  v27 = 0x300000001LL;
  if ( !byte_4FDE388 && (unsigned int)sub_2207590((__int64)&byte_4FDE388) )
  {
    sub_23B0820((__int64 *)&v20, byte_3F871B3);
    qword_4FDE3A0 = (__int64)&qword_4FDE3B0;
    qword_4FDE3A8 = 0x100000000LL;
    qword_4FDE3B0 = (__int64)&unk_4FDE3C0;
    sub_23AEDD0(&qword_4FDE3B0, v20, (__int64)&v20[v21]);
    LODWORD(qword_4FDE3A8) = qword_4FDE3A8 + 1;
    __cxa_atexit((void (*)(void *))sub_BC5500, &qword_4FDE3A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FDE388);
    sub_2240A30((unsigned __int64 *)&v20);
  }
  v8 = v26;
  v9 = &qword_4FDE3E0;
  if ( (unsigned int)sub_BC7640((__int64)&qword_4FDE3E0, (__int64)v26, (unsigned int)v27, (__int64)&qword_4FDE3A0) )
  {
    v15 = sub_C5F790((__int64)&qword_4FDE3E0, (__int64)v8);
    sub_904010(v15, "Unable to create temporary file.");
  }
  else
  {
    if ( !byte_4FDE340 )
    {
      v9 = (__int64 *)&byte_4FDE340;
      if ( (unsigned int)sub_2207590((__int64)&byte_4FDE340) )
      {
        sub_C86E60((char *)&qword_4FDE360, qword_4FDE4A8, qword_4FDE4B0, 0, 0);
        v8 = &qword_4FDE360;
        __cxa_atexit((void (*)(void *))sub_BC5B10, &qword_4FDE360, &qword_4A427C0);
        v9 = (__int64 *)&byte_4FDE340;
        sub_2207640((__int64)&byte_4FDE340);
      }
    }
    if ( (byte_4FDE380 & 1) != 0 )
    {
      v14 = sub_C5F790((__int64)v9, (__int64)v8);
      sub_904010(v14, "Unable to find test-changed executable.");
    }
    else
    {
      v10 = qword_4FDE360;
      v11 = qword_4FDE368;
      v20 = (_BYTE *)qword_4FDE4A8;
      v21 = qword_4FDE4B0;
      v22 = *(_QWORD *)qword_4FDE3A0;
      v23 = *(_QWORD *)(qword_4FDE3A0 + 8);
      v24 = a3;
      v25 = a4;
      if ( (int)sub_C881F0((_BYTE *)qword_4FDE360, qword_4FDE368, &v20, 3, 0, 0, v18, v19, 0, 0, 0, 0, 0, 0) < 0 )
      {
        v16 = sub_C5F790(v10, v11);
        sub_904010(v16, "Error executing test-changed executable.");
      }
      else
      {
        v12 = (unsigned int)qword_4FDE3A8;
        v13 = qword_4FDE3A0;
        if ( (unsigned int)sub_BC5E90(qword_4FDE3A0, (unsigned int)qword_4FDE3A8) )
        {
          v17 = sub_C5F790(v13, v12);
          sub_904010(v17, "Unable to remove temporary file.");
        }
      }
    }
  }
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
}
