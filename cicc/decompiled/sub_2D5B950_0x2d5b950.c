// Function: sub_2D5B950
// Address: 0x2d5b950
//
__int64 __fastcall sub_2D5B950(
        __int64 *a1,
        char a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        int a5,
        char a6,
        __int64 a7,
        __int64 a8)
{
  __int64 (*v10)(void); // rax
  __int64 v11; // r12
  int v13; // r14d
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  _BYTE v20[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v10 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 16LL);
  if ( (char *)v10 != (char *)sub_9202E0 )
  {
    v11 = v10();
    goto LABEL_6;
  }
  if ( *a3 <= 0x15u && *a4 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(a2) )
      v11 = sub_AD5570(a2, (__int64)a3, a4, 0, 0);
    else
      v11 = sub_AABE40(a2, a3, a4);
LABEL_6:
    if ( v11 )
      return v11;
  }
  v21 = 257;
  v11 = sub_B504D0(a2, (__int64)a3, (__int64)a4, (__int64)v20, 0, 0);
  if ( (unsigned __int8)sub_920620(v11) )
  {
    v13 = a5;
    if ( !a6 )
      v13 = *((_DWORD *)a1 + 26);
    if ( a8 || (a8 = a1[12]) != 0 )
      sub_B99FD0(v11, 3u, a8);
    sub_B45150(v11, v13);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a7,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v15 )
  {
    do
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v11, v17, v16);
    }
    while ( v15 != v14 );
  }
  return v11;
}
