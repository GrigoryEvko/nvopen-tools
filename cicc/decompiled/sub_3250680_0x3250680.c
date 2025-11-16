// Function: sub_3250680
// Address: 0x3250680
//
unsigned __int8 *__fastcall sub_3250680(_QWORD *a1, unsigned __int8 *a2, char a3)
{
  __int64 v3; // r14
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned __int8 *v8; // r13
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax

  v3 = (__int64)(a1 + 1);
  if ( !a3 )
  {
    v6 = *(a2 - 16);
    if ( (v6 & 2) != 0 )
      v7 = *((_QWORD *)a2 - 4);
    else
      v7 = (__int64)&a2[-8 * ((v6 >> 2) & 0xF) - 16];
    v3 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*a1 + 48LL))(a1, *(_QWORD *)(v7 + 8));
  }
  v8 = sub_3247C80((__int64)a1, a2);
  if ( !v8 )
  {
    v10 = *(a2 - 16);
    if ( (v10 & 2) != 0 )
      v11 = *((_QWORD *)a2 - 4);
    else
      v11 = (__int64)&a2[-8 * ((v10 >> 2) & 0xF) - 16];
    v12 = *(_QWORD *)(v11 + 48);
    if ( v12 && !a3 )
    {
      v3 = (__int64)(a1 + 1);
      sub_3250680(a1, v12, 0);
    }
    v13 = sub_324C6D0(a1, 46, v3, a2);
    v8 = (unsigned __int8 *)v13;
    if ( (a2[36] & 8) == 0 )
    {
      v14 = sub_3215100(v13);
      sub_324FBC0(v14, (__int64)a2, (unsigned __int64)v8, 0);
    }
  }
  return v8;
}
