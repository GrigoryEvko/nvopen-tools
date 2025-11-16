// Function: sub_D951F0
// Address: 0xd951f0
//
__int64 __fastcall sub_D951F0(__int64 a1, __int64 a2, unsigned int a3)
{
  const void **v3; // r8
  unsigned int v5; // r15d
  unsigned int v6; // eax
  unsigned __int64 v8; // rax
  const void **v9; // r8
  bool v10; // cc
  unsigned __int64 v11; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-58h]
  unsigned __int64 v13; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-48h]
  __int64 v15; // [rsp+30h] [rbp-40h] BYREF
  int v16; // [rsp+38h] [rbp-38h]

  v3 = (const void **)(a2 + 16);
  v5 = *(_DWORD *)(a2 + 8);
  if ( a3 <= v5 )
  {
    if ( a3 >= v5 )
    {
      *(_DWORD *)(a1 + 8) = v5;
      if ( v5 > 0x40 )
      {
        sub_C43780(a1, (const void **)a2);
        v3 = (const void **)(a2 + 16);
      }
      else
      {
        *(_QWORD *)a1 = *(_QWORD *)a2;
      }
      v6 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v6;
      if ( v6 > 0x40 )
        sub_C43780(a1 + 16, v3);
      else
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
    }
    sub_C44740((__int64)&v15, (char **)(a2 + 16), a3);
    sub_C44740((__int64)&v13, (char **)a2, a3);
    *(_DWORD *)(a1 + 8) = v14;
    v8 = v13;
LABEL_9:
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 24) = v16;
    *(_QWORD *)(a1 + 16) = v15;
    return a1;
  }
  sub_C449B0((__int64)&v11, (const void **)a2, a3);
  v9 = (const void **)(a2 + 16);
  if ( v5 != v12 )
  {
    if ( v5 > 0x3F || v12 > 0x40 )
    {
      sub_C43C90(&v11, v5, v12);
      v9 = (const void **)(a2 + 16);
    }
    else
    {
      v11 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 - (unsigned __int8)v12 + 64) << v5;
    }
  }
  sub_C449B0((__int64)&v15, v9, a3);
  v14 = v12;
  if ( v12 <= 0x40 )
  {
    *(_DWORD *)(a1 + 8) = v12;
    v8 = v11;
    goto LABEL_9;
  }
  sub_C43780((__int64)&v13, (const void **)&v11);
  v10 = v12 <= 0x40;
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 16) = v15;
  if ( !v10 && v11 )
    j_j___libc_free_0_0(v11);
  return a1;
}
