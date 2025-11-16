// Function: sub_3735570
// Address: 0x3735570
//
__int64 __fastcall sub_3735570(__int64 a1, unsigned int a2, unsigned __int64 **a3)
{
  _BYTE **v4; // rbx
  __int64 v5; // rdx
  int v6; // eax
  __int64 *v8; // rsi
  unsigned __int64 v9; // rsi
  __int64 v10; // rsi
  unsigned __int64 v11; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-18h]

  v4 = *(_BYTE ***)a1;
  v5 = *(_QWORD *)(**(_QWORD **)(a1 + 8) + 8LL) + 24LL * a2;
  v6 = *(_DWORD *)v5;
  if ( !*(_DWORD *)v5 )
    return sub_3243770((__int64)*v4, v4[1], a3, *(_DWORD *)(v5 + 20));
  if ( v6 == 1 )
  {
    sub_3243300((__int64)*v4, *(_QWORD *)(v5 + 8));
    return 1;
  }
  if ( v6 != 2 )
  {
    if ( v6 != 3 )
    {
      if ( v6 != 4 )
        BUG();
      sub_3243F80(*v4, *(_DWORD *)(v5 + 16), *(int *)(v5 + 20));
      return 1;
    }
    v10 = *(_QWORD *)(v5 + 8);
    v12 = *(_DWORD *)(v10 + 32);
    if ( v12 <= 0x40 )
    {
      v11 = *(_QWORD *)(v10 + 24);
LABEL_18:
      v9 = v11;
      goto LABEL_12;
    }
    sub_C43780((__int64)&v11, (const void **)(v10 + 24));
    if ( v12 <= 0x40 )
      goto LABEL_18;
    if ( v11 )
      j_j___libc_free_0_0(v11);
    return 0;
  }
  v8 = (__int64 *)(*(_QWORD *)(v5 + 8) + 24LL);
  if ( (void *)*v8 == sub_C33340() )
    sub_C3E660((__int64)&v11, (__int64)v8);
  else
    sub_C3A850((__int64)&v11, v8);
  v9 = v11;
  if ( v12 <= 0x40 )
  {
LABEL_12:
    sub_3243300((__int64)*v4, v9);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    return 1;
  }
  if ( !v11 )
    return 0;
  j_j___libc_free_0_0(v11);
  return 0;
}
