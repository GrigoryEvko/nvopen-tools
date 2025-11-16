// Function: sub_11DBC90
// Address: 0x11dbc90
//
unsigned __int8 *__fastcall sub_11DBC90(__int64 *a1, unsigned __int8 **a2, unsigned __int64 a3)
{
  unsigned __int8 *v3; // r15
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // r14
  __int64 (__fastcall *v10)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // esi
  _BYTE v18[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v19; // [rsp+40h] [rbp-70h]
  _BYTE v20[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v21; // [rsp+70h] [rbp-40h]

  v3 = *a2;
  if ( a3 > 1 )
  {
    v5 = 1;
    v6 = 1;
    while ( 1 )
    {
      v8 = a1[10];
      v19 = 257;
      v9 = a2[v6];
      v10 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v8 + 16LL);
      if ( v10 != sub_9202E0 )
        break;
      if ( *v3 > 0x15u || *v9 > 0x15u )
      {
LABEL_11:
        v21 = 257;
        v3 = (unsigned __int8 *)sub_B504D0(29, (__int64)v3, (__int64)v9, (__int64)v20, 0, 0);
        (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
          a1[11],
          v3,
          v18,
          a1[7],
          a1[8]);
        v11 = *a1;
        v12 = *a1 + 16LL * *((unsigned int *)a1 + 2);
        if ( *a1 == v12 )
          goto LABEL_8;
        do
        {
          v13 = *(_QWORD *)(v11 + 8);
          v14 = *(_DWORD *)v11;
          v11 += 16;
          sub_B99FD0((__int64)v3, v14, v13);
        }
        while ( v12 != v11 );
        v6 = ++v5;
        if ( v5 >= a3 )
          return v3;
      }
      else
      {
        if ( (unsigned __int8)sub_AC47B0(29) )
          v7 = sub_AD5570(29, (__int64)v3, v9, 0, 0);
        else
          v7 = sub_AABE40(0x1Du, v3, v9);
LABEL_6:
        if ( !v7 )
          goto LABEL_11;
        v3 = (unsigned __int8 *)v7;
LABEL_8:
        v6 = ++v5;
        if ( v5 >= a3 )
          return v3;
      }
    }
    v7 = v10(v8, 29u, v3, v9);
    goto LABEL_6;
  }
  return v3;
}
