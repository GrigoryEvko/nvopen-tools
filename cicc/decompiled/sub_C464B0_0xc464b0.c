// Function: sub_C464B0
// Address: 0xc464b0
//
__int64 __fastcall sub_C464B0(__int64 a1, __int64 a2, signed __int64 a3)
{
  unsigned int v5; // edx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rbx
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // rdi
  unsigned int v10; // eax
  unsigned __int64 *v11; // rax
  bool v12; // cc
  unsigned __int64 *v13; // rdi
  unsigned __int64 *v15; // rdi
  unsigned int v16; // eax
  unsigned __int64 *v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]
  unsigned __int64 v21; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-28h]

  v5 = *(_DWORD *)(a2 + 8);
  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  if ( v5 > 0x40 )
    v7 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
  v8 = (unsigned __int64 *)((1LL << ((unsigned __int8)v5 - 1)) & v7);
  if ( v8 )
  {
    if ( a3 >= 0 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
      {
        sub_C43780((__int64)&v17, (const void **)a2);
        v5 = v18;
        if ( v18 > 0x40 )
        {
          sub_C43D10((__int64)&v17);
          goto LABEL_9;
        }
        v6 = (unsigned __int64)v17;
      }
      v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6;
      if ( !v5 )
        v9 = 0;
      v17 = (unsigned __int64 *)v9;
LABEL_9:
      sub_C46250((__int64)&v17);
      v10 = v18;
      v18 = 0;
      v20 = v10;
      v19 = v17;
      sub_C45850((__int64)&v21, &v19, a3);
      if ( v22 > 0x40 )
      {
        sub_C43D10((__int64)&v21);
      }
      else
      {
        v11 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v21);
        if ( !v22 )
          v11 = 0;
        v21 = (unsigned __int64)v11;
      }
      sub_C46250((__int64)&v21);
      v12 = v20 <= 0x40;
      *(_DWORD *)(a1 + 8) = v22;
      *(_QWORD *)a1 = v21;
      if ( !v12 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v18 > 0x40 )
      {
        v13 = v17;
        if ( v17 )
        {
LABEL_18:
          j_j___libc_free_0_0(v13);
          return a1;
        }
      }
      return a1;
    }
    v20 = *(_DWORD *)(a2 + 8);
    if ( v5 > 0x40 )
    {
      sub_C43780((__int64)&v19, (const void **)a2);
      v5 = v20;
      if ( v20 > 0x40 )
      {
        sub_C43D10((__int64)&v19);
        goto LABEL_23;
      }
      v6 = (unsigned __int64)v19;
    }
    v15 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6);
    if ( !v5 )
      v15 = 0;
    v19 = v15;
LABEL_23:
    sub_C46250((__int64)&v19);
    v16 = v20;
    v20 = 0;
    v22 = v16;
    v21 = (unsigned __int64)v19;
    sub_C45850(a1, (unsigned __int64 **)&v21, -a3);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 )
    {
      v13 = v19;
      if ( v19 )
        goto LABEL_18;
    }
    return a1;
  }
  if ( a3 >= 0 )
  {
    sub_C45850(a1, (unsigned __int64 **)a2, a3);
    return a1;
  }
  sub_C45850((__int64)&v21, (unsigned __int64 **)a2, -a3);
  if ( v22 <= 0x40 )
  {
    if ( v22 )
      v8 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & ~v21);
    v21 = (unsigned __int64)v8;
  }
  else
  {
    sub_C43D10((__int64)&v21);
  }
  sub_C46250((__int64)&v21);
  *(_DWORD *)(a1 + 8) = v22;
  *(_QWORD *)a1 = v21;
  return a1;
}
