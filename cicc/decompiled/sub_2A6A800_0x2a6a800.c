// Function: sub_2A6A800
// Address: 0x2a6a800
//
void __fastcall sub_2A6A800(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int64 *v3; // r14
  unsigned __int8 *v4; // rax
  unsigned __int8 *v5; // rax
  __int64 v6; // r9
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int8 v8[80]; // [rsp+10h] [rbp-50h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15 )
  {
    sub_2A6A450(a1, a2);
  }
  else
  {
    v2 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 32));
    sub_22C05A0((__int64)v8, v2);
    v7 = a2;
    v3 = sub_2A686D0(a1 + 136, &v7);
    if ( *(_BYTE *)v3 == 6 )
    {
      sub_2A6A450(a1, a2);
    }
    else if ( v8[0] > 1u )
    {
      if ( !(unsigned __int8)sub_2A62D90((__int64)v8)
        || (v4 = (unsigned __int8 *)sub_2A637C0(a1, (__int64)v8, *(_QWORD *)(a2 + 8)), !sub_98ED60(v4, 0, 0, 0, 0)) )
      {
        sub_2A6A450(a1, a2);
        sub_22C0090(v8);
        return;
      }
      v5 = (unsigned __int8 *)sub_2A637C0(a1, (__int64)v8, *(_QWORD *)(a2 + 8));
      sub_2A63320(a1, (__int64)v3, a2, v5, 0, v6);
    }
    sub_22C0090(v8);
  }
}
