// Function: sub_2566CF0
// Address: 0x2566cf0
//
void __fastcall sub_2566CF0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  unsigned __int64 v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-48h]
  unsigned __int64 v11; // [rsp+10h] [rbp-40h]
  unsigned int v12; // [rsp+18h] [rbp-38h]

  v2 = (_QWORD *)(a1 + 72);
  if ( sub_2566C40(a2 + 32, (__int64 *)(a1 + 72)) )
  {
    if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v3 = *(_DWORD *)(a1 + 144), v3 <= 0x40) )
    {
      v8 = *(_QWORD *)(a1 + 136);
      *(_DWORD *)(a1 + 112) = v3;
      *(_QWORD *)(a1 + 104) = v8;
    }
    else
    {
      sub_C43990(a1 + 104, a1 + 136);
    }
    if ( *(_DWORD *)(a1 + 128) <= 0x40u && (v4 = *(_DWORD *)(a1 + 160), v4 <= 0x40) )
    {
      v7 = *(_QWORD *)(a1 + 152);
      *(_DWORD *)(a1 + 128) = v4;
      *(_QWORD *)(a1 + 120) = v7;
    }
    else
    {
      sub_C43990(a1 + 120, a1 + 152);
    }
  }
  else
  {
    v5 = sub_2509740(v2);
    sub_254EAA0((__int64)&v9, a1, a2, v5);
    sub_254F8E0(a1 + 88, (__int64)&v9);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    v6 = sub_2509740(v2);
    sub_254EE20((__int64)&v9, a1, a2, v6);
    sub_254F8E0(a1 + 88, (__int64)&v9);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
  }
}
