// Function: sub_25550E0
// Address: 0x25550e0
//
__int64 __fastcall sub_25550E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // eax
  _BYTE *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-B8h]
  __int64 v17; // [rsp+8h] [rbp-B8h]
  __int64 v18[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v20[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v21; // [rsp+40h] [rbp-80h] BYREF
  __int64 v22; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+58h] [rbp-68h]
  __int64 v24; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+68h] [rbp-58h]
  __int64 v26[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v27[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( !a4
    || a4 == sub_2509740((_QWORD *)(a2 + 72))
    || (v16 = sub_B43CB0(a4), v7 = sub_250D070((_QWORD *)(a2 + 72)), !(unsigned __int8)sub_250C180(v7, v16))
    || (v11 = (_BYTE *)sub_250D070((_QWORD *)(a2 + 72)), v12 = (__int64)v11, *v11 > 0x1Cu)
    && ((v17 = *(_QWORD *)(a3 + 208),
         v13 = sub_B43CB0((__int64)v11),
         (v14 = sub_2554D30(*(_QWORD *)(v17 + 240), v13, 0)) == 0)
     || !(unsigned __int8)sub_B19DB0(v14, v12, a4)) )
  {
    v8 = *(_DWORD *)(a2 + 112);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
    {
      sub_C43780(a1, (const void **)(a2 + 104));
      v15 = *(_DWORD *)(a2 + 128);
      *(_DWORD *)(a1 + 24) = v15;
      if ( v15 <= 0x40 )
        goto LABEL_6;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)(a2 + 104);
      v9 = *(_DWORD *)(a2 + 128);
      *(_DWORD *)(a1 + 24) = v9;
      if ( v9 <= 0x40 )
      {
LABEL_6:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 120);
        return a1;
      }
    }
    sub_C43780(a1 + 16, (const void **)(a2 + 120));
    return a1;
  }
  sub_254EE20((__int64)v18, a2, a3, a4);
  sub_254EAA0((__int64)v20, a2, a3, a4);
  v23 = *(_DWORD *)(a2 + 112);
  if ( v23 > 0x40 )
    sub_C43780((__int64)&v22, (const void **)(a2 + 104));
  else
    v22 = *(_QWORD *)(a2 + 104);
  v25 = *(_DWORD *)(a2 + 128);
  if ( v25 > 0x40 )
    sub_C43780((__int64)&v24, (const void **)(a2 + 120));
  else
    v24 = *(_QWORD *)(a2 + 120);
  sub_AB2160((__int64)v26, (__int64)&v22, (__int64)v20, 0);
  sub_AB2160(a1, (__int64)v26, (__int64)v18, 0);
  sub_969240(v27);
  sub_969240(v26);
  sub_969240(&v24);
  sub_969240(&v22);
  sub_969240(&v21);
  sub_969240(v20);
  sub_969240(&v19);
  sub_969240(v18);
  return a1;
}
