// Function: sub_2A6B210
// Address: 0x2a6b210
//
void __fastcall sub_2A6B210(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  unsigned __int8 *v3; // rax
  unsigned __int8 *v4; // rax
  unsigned __int8 *v5; // rax
  __int64 *v6; // rax
  unsigned __int8 v7[48]; // [rsp+10h] [rbp-F0h] BYREF
  unsigned __int8 v8[48]; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 v9[48]; // [rsp+70h] [rbp-90h] BYREF
  __int64 v10[12]; // [rsp+A0h] [rbp-60h] BYREF

  v10[0] = a2;
  if ( *(_BYTE *)sub_2A686D0((__int64)(a1 + 17), v10) == 6 )
  {
    sub_2A6A450((__int64)a1, a2);
  }
  else
  {
    v2 = *(unsigned __int8 **)(a2 - 32);
    v3 = (unsigned __int8 *)sub_2A68BC0((__int64)a1, *(unsigned __int8 **)(a2 - 64));
    sub_22C05A0((__int64)v7, v3);
    v4 = (unsigned __int8 *)sub_2A68BC0((__int64)a1, v2);
    sub_22C05A0((__int64)v8, v4);
    v5 = (unsigned __int8 *)sub_22EAB60((char *)v7, *(_WORD *)(a2 + 2) & 0x3F, *(_QWORD *)(a2 + 8), (char *)v8, *a1);
    if ( v5 )
    {
      *(_QWORD *)v9 = 0;
      sub_2A624B0((__int64)v9, v5, 0);
      sub_22C05A0((__int64)v10, v9);
      sub_2A689D0((__int64)a1, a2, (unsigned __int8 *)v10, 0x100000000LL);
      sub_22C0090((unsigned __int8 *)v10);
      sub_22C0090(v9);
      sub_22C0090(v8);
      sub_22C0090(v7);
    }
    else if ( v7[0] > 1u && v8[0] > 1u
           || (v10[0] = a2, v6 = sub_2A686D0((__int64)(a1 + 17), v10), (unsigned __int8)sub_2A62D90((__int64)v6)) )
    {
      sub_2A6A450((__int64)a1, a2);
      sub_22C0090(v8);
      sub_22C0090(v7);
    }
    else
    {
      sub_22C0090(v8);
      sub_22C0090(v7);
    }
  }
}
