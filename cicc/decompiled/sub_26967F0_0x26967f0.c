// Function: sub_26967F0
// Address: 0x26967f0
//
void __fastcall sub_26967F0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned __int8 v5; // al
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_QWORD *)(a1 + 72);
  v3 = v2 & 3;
  v4 = v2 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v3 == 3 )
    v4 = *(_QWORD *)(v4 + 24);
  v5 = *(_BYTE *)v4;
  if ( *(_BYTE *)v4 )
  {
    if ( v5 == 22 )
    {
      v4 = *(_QWORD *)(v4 + 24);
    }
    else if ( v5 <= 0x1Cu )
    {
      v4 = 0;
    }
    else
    {
      v4 = sub_B43CB0(v4);
    }
  }
  v8[0] = v4;
  v6 = (_QWORD *)sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
  {
    *v6 = v6 + 2;
    v6[1] = 0x800000000LL;
    sub_2696480((__int64)v6, v8);
  }
  *(_QWORD *)(a1 + 464) = v7;
}
