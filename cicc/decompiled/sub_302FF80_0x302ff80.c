// Function: sub_302FF80
// Address: 0x302ff80
//
__int64 __fastcall sub_302FF80(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int16 v4; // bx
  __int64 *v5; // rsi
  __int64 v7; // rax
  unsigned int v8; // edx
  char *v9; // rax
  char *v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]
  char *v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 40) + 40LL * a3;
  if ( *(_DWORD *)(*(_QWORD *)v3 + 24LL) == 51 )
  {
    *(_DWORD *)(a1 + 8) = 32;
    *(_QWORD *)a1 = 0;
    return a1;
  }
  else
  {
    v4 = **(_WORD **)(a2 + 48);
    v11 = 1;
    v10 = 0;
    if ( v4 == 127 || v4 == 138 )
    {
      v5 = (__int64 *)(*(_QWORD *)(*(_QWORD *)v3 + 96LL) + 24LL);
      if ( (void *)*v5 == sub_C33340() )
        sub_C3E660((__int64)&v12, (__int64)v5);
      else
        sub_C3A850((__int64)&v12, v5);
      v10 = v12;
      v11 = v13;
    }
    else
    {
      if ( v4 != 47 && v4 != 37 )
        BUG();
      v7 = *(_QWORD *)(*(_QWORD *)v3 + 96LL);
      v8 = *(_DWORD *)(v7 + 32);
      if ( v8 <= 0x40 )
      {
        v9 = *(char **)(v7 + 24);
        v11 = v8;
        v10 = v9;
      }
      else
      {
        sub_C43990((__int64)&v10, v7 + 24);
      }
    }
    if ( v4 == 37 )
    {
      sub_C44740((__int64)&v12, &v10, 8u);
      if ( v11 > 0x40 && v10 )
        j_j___libc_free_0_0((unsigned __int64)v10);
      v10 = v12;
      v11 = v13;
    }
    sub_C449B0(a1, (const void **)&v10, 0x20u);
    if ( v11 > 0x40 )
    {
      if ( v10 )
        j_j___libc_free_0_0((unsigned __int64)v10);
    }
    return a1;
  }
}
