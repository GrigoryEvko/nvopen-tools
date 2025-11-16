// Function: sub_2589870
// Address: 0x2589870
//
__int64 __fastcall sub_2589870(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned int v8; // r14d
  int v9; // eax
  int v11; // eax
  __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  int v13; // [rsp+8h] [rbp-58h]
  unsigned __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-38h]

  v4 = sub_250D2C0(a2, 0);
  v6 = sub_2589400(*(_QWORD *)a1, v4, v5, *(_QWORD *)(a1 + 8), (unsigned int)(**(_BYTE **)(a1 + 16) == 0) + 1, 0, 1);
  if ( v6 )
  {
    v7 = v6;
    if ( **(_BYTE **)(a1 + 16) )
    {
      v15 = *(_DWORD *)(v6 + 112);
      if ( v15 > 0x40 )
        sub_C43780((__int64)&v14, (const void **)(v6 + 104));
      else
        v14 = *(_QWORD *)(v6 + 104);
      v17 = *(_DWORD *)(v7 + 128);
      if ( v17 > 0x40 )
        sub_C43780((__int64)&v16, (const void **)(v7 + 120));
      else
        v16 = *(_QWORD *)(v7 + 120);
    }
    else
    {
      v15 = *(_DWORD *)(v6 + 144);
      if ( v15 > 0x40 )
        sub_C43780((__int64)&v14, (const void **)(v6 + 136));
      else
        v14 = *(_QWORD *)(v6 + 136);
      v17 = *(_DWORD *)(v7 + 160);
      if ( v17 > 0x40 )
        sub_C43780((__int64)&v16, (const void **)(v7 + 152));
      else
        v16 = *(_QWORD *)(v7 + 152);
    }
    v8 = 0;
    if ( !sub_AAF760((__int64)&v14) )
    {
      v8 = **(unsigned __int8 **)(a1 + 24);
      if ( (_BYTE)v8 )
      {
        sub_AB14C0((__int64)&v12, (__int64)&v14);
        if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
          j_j___libc_free_0_0(*(_QWORD *)a3);
        *(_QWORD *)a3 = v12;
        v9 = v13;
        v13 = 0;
        *(_DWORD *)(a3 + 8) = v9;
        sub_969240(&v12);
      }
      else
      {
        sub_AB13A0((__int64)&v12, (__int64)&v14);
        if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
          j_j___libc_free_0_0(*(_QWORD *)a3);
        v8 = 1;
        *(_QWORD *)a3 = v12;
        v11 = v13;
        v13 = 0;
        *(_DWORD *)(a3 + 8) = v11;
        sub_969240(&v12);
      }
    }
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  else
  {
    return 0;
  }
  return v8;
}
