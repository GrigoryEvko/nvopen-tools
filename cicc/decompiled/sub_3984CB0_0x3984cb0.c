// Function: sub_3984CB0
// Address: 0x3984cb0
//
__int64 __fastcall sub_3984CB0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  void *v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-54h]
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  unsigned __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h]

  v6 = *a3;
  v17 = 0;
  v18 = 0;
  if ( v6 )
  {
    v17 = *(_QWORD *)(v6 + 24);
    v18 = *(_QWORD *)(v6 + 32);
  }
  sub_399FD50(a4, v6);
  v7 = *((_DWORD *)a3 + 2);
  if ( v7 == 1 )
  {
    if ( a2 && (unsigned int)(*(_DWORD *)(a2 + 52) - 5) <= 1 )
      sub_399F630(a4, a3[2]);
    else
      sub_399F670(a4, a3[2]);
    return sub_399FAC0(a4, &v17, 0);
  }
  if ( v7 )
  {
    if ( v7 == 2 )
    {
      v13 = sub_16982C0();
      v14 = (__int64 *)(a3[2] + 32);
      if ( (void *)*v14 == v13 )
        sub_169D930((__int64)&v19, (__int64)v14);
      else
        sub_169D7E0((__int64)&v19, v14);
      sub_399F6B0(a4, &v19);
      if ( (unsigned int)v20 > 0x40 )
      {
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
    }
    return sub_399FAC0(a4, &v17, 0);
  }
  v9 = *((unsigned int *)a3 + 7);
  if ( !*((_BYTE *)a3 + 24) )
    *(_DWORD *)(a4 + 76) = 2;
  v19 = 0;
  v20 = 0;
  if ( v6 )
  {
    v19 = *(_QWORD *)(v6 + 24);
    v20 = *(_QWORD *)(v6 + 32);
  }
  v10 = 0;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 264) + 16LL);
  v12 = *(__int64 (**)())(*(_QWORD *)v11 + 112LL);
  if ( v12 != sub_1D00B10 )
  {
    v16 = v9;
    v15 = ((__int64 (__fastcall *)(__int64, _QWORD))v12)(v11, 0);
    v9 = v16;
    v10 = v15;
  }
  result = sub_399F750(a4, v10, &v19, v9, 0);
  if ( (_BYTE)result )
    return sub_399FAC0(a4, &v19, 0);
  return result;
}
