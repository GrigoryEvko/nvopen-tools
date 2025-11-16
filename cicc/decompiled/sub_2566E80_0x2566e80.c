// Function: sub_2566E80
// Address: 0x2566e80
//
void __fastcall sub_2566E80(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  _BYTE *v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // eax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = (__int64 *)(a1 + 72);
  v4 = (_BYTE *)sub_250D070((_QWORD *)(a1 + 72));
  if ( *v4 == 85 && (v4[7] & 0x20) != 0 )
  {
    v11 = sub_B91C10((__int64)v4, 4);
    if ( v11 )
    {
      sub_ABEA30((__int64)v12, v11);
      sub_254F8E0(a1 + 88, (__int64)v12);
      sub_969240(v13);
      sub_969240(v12);
    }
  }
  if ( sub_2566C40(a2 + 32, v2) )
  {
    if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v5 = *(_DWORD *)(a1 + 144), v5 <= 0x40) )
    {
      v10 = *(_QWORD *)(a1 + 136);
      *(_DWORD *)(a1 + 112) = v5;
      *(_QWORD *)(a1 + 104) = v10;
    }
    else
    {
      sub_C43990(a1 + 104, a1 + 136);
    }
    if ( *(_DWORD *)(a1 + 128) <= 0x40u && (v6 = *(_DWORD *)(a1 + 160), v6 <= 0x40) )
    {
      v9 = *(_QWORD *)(a1 + 152);
      *(_DWORD *)(a1 + 128) = v6;
      *(_QWORD *)(a1 + 120) = v9;
    }
    else
    {
      sub_C43990(a1 + 120, a1 + 152);
    }
  }
  else
  {
    v7 = sub_2509740(v2);
    sub_254EAA0((__int64)v12, a1, a2, v7);
    sub_254F8E0(a1 + 88, (__int64)v12);
    sub_969240(v13);
    sub_969240(v12);
    v8 = sub_2509740(v2);
    sub_254EE20((__int64)v12, a1, a2, v8);
    sub_254F8E0(a1 + 88, (__int64)v12);
    sub_969240(v13);
    sub_969240(v12);
  }
}
