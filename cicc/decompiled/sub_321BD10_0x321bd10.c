// Function: sub_321BD10
// Address: 0x321bd10
//
__int64 __fastcall sub_321BD10(_QWORD *a1, unsigned int a2, _QWORD *a3)
{
  __int64 v5; // r12
  int v6; // eax
  unsigned int v7; // r14d
  unsigned int v9; // r14d
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rax
  void *v16; // r13
  __int64 *v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // r15
  unsigned __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a1[3] + 8LL) + 24LL * a2;
  v6 = *(_DWORD *)v5;
  if ( *(_DWORD *)v5 == 1 )
  {
    v12 = (__int64 *)a1[1];
    v13 = *(_QWORD *)(v5 + 8);
    v14 = *a1;
    v15 = *v12;
    if ( v15 && (unsigned int)(*(_DWORD *)(v15 + 44) - 5) <= 1 )
    {
      sub_32432C0(v14, v13);
      return 1;
    }
    else
    {
      sub_3243300(v14, v13);
      return 1;
    }
  }
  if ( v6 )
  {
    if ( v6 == 4 )
    {
      v7 = 1;
      sub_3243F80(*a1, *(unsigned int *)(v5 + 16), *(int *)(v5 + 20));
    }
    else
    {
      v7 = 1;
      if ( v6 == 2 )
      {
        if ( (unsigned __int16)sub_31DF670(a1[2]) > 3u
          && *(_DWORD *)(*(_QWORD *)(a1[2] + 760LL) + 6224LL) != 3
          && *a3 == a3[1] )
        {
          sub_32433D0(*a1, *(_QWORD *)(v5 + 8) + 24LL);
        }
        else
        {
          v16 = sub_C33340();
          v17 = (__int64 *)(*(_QWORD *)(v5 + 8) + 24LL);
          if ( (void *)*v17 == v16 )
            sub_C3E660((__int64)&v20, (__int64)v17);
          else
            sub_C3A850((__int64)&v20, v17);
          if ( v21 <= 0x40 )
          {
            v18 = *(_QWORD *)(v5 + 8);
            v19 = *a1;
            if ( v16 == *(void **)(v18 + 24) )
              sub_C3E660((__int64)&v20, v18 + 24);
            else
              sub_C3A850((__int64)&v20, (__int64 *)(v18 + 24));
            sub_3243320(v19, &v20);
            if ( v21 > 0x40 && v20 )
              j_j___libc_free_0_0(v20);
            return 1;
          }
          else
          {
            if ( v20 )
              j_j___libc_free_0_0(v20);
            return 0;
          }
        }
      }
    }
    return v7;
  }
  v9 = *(_DWORD *)(v5 + 20);
  if ( !*(_BYTE *)(v5 + 16) )
    *(_BYTE *)(*a1 + 100LL) = *(_BYTE *)(*a1 + 100LL) & 0xF8 | 2;
  v10 = *(_QWORD *)(*(_QWORD *)(a1[2] + 232LL) + 16LL);
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 200LL))(v10);
  return sub_3243770(*a1, v11, a3, v9, 0);
}
