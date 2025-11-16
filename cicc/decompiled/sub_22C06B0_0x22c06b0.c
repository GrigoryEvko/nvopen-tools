// Function: sub_22C06B0
// Address: 0x22c06b0
//
__int64 __fastcall sub_22C06B0(__int64 a1, __int64 a2, char a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-68h]
  unsigned __int64 v9; // [rsp+10h] [rbp-60h]
  unsigned int v10; // [rsp+18h] [rbp-58h]
  unsigned __int8 v11[80]; // [rsp+20h] [rbp-50h] BYREF

  if ( sub_AAF760(a2) )
  {
    *(_WORD *)a1 = 6;
    return a1;
  }
  else if ( sub_AAF7D0(a2) )
  {
    *(_BYTE *)a1 = a3;
    *(_BYTE *)(a1 + 1) = 0;
    *(_QWORD *)v11 = 0;
    sub_22C0090(v11);
    return a1;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a2 + 8) = 0;
    v8 = v5;
    v6 = *(_QWORD *)a2;
    *(_QWORD *)v11 = 0;
    v7 = v6;
    LODWORD(v6) = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a2 + 24) = 0;
    v10 = v6;
    v9 = *(_QWORD *)(a2 + 16);
    sub_22C00F0((__int64)v11, (__int64)&v7, a3, 0, 1u);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    if ( v8 > 0x40 )
    {
      if ( v7 )
        j_j___libc_free_0_0(v7);
    }
    sub_22C0650(a1, v11);
    sub_22C0090(v11);
    return a1;
  }
}
