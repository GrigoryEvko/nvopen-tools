// Function: sub_ABD110
// Address: 0xabd110
//
__int64 __fastcall sub_ABD110(__int64 a1, __int64 a2, char a3)
{
  unsigned int v5; // r15d
  unsigned int v6; // ebx
  unsigned int v7; // r14d
  __int64 v8; // r9
  int v9; // eax
  __int64 v10; // rax
  int v11; // eax
  unsigned int v12; // ebx
  __int64 v13; // r9
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-98h]
  __int64 v17; // [rsp+8h] [rbp-98h]
  __int64 v18; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-88h]
  __int64 v20; // [rsp+20h] [rbp-80h] BYREF
  int v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+30h] [rbp-70h] BYREF
  int v23; // [rsp+38h] [rbp-68h]
  __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+58h] [rbp-48h]
  __int64 v27[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  v5 = *(_DWORD *)(a2 + 8);
  sub_9691E0((__int64)&v18, v5, 0, 0, 0);
  if ( !a3 || !sub_AB1B10(a2, (__int64)&v18) )
  {
    if ( sub_AAF760(a2) )
    {
      sub_9691E0((__int64)&v20, v5, v5, 0, 0);
      sub_C46A40(&v20, 1);
      v11 = v21;
      v21 = 0;
      v23 = v11;
      v22 = v20;
      v26 = v19;
      if ( v19 > 0x40 )
        sub_C43780(&v25, &v18);
      else
        v25 = v18;
      sub_9875E0(a1, &v25, &v22);
      sub_969240(&v25);
      sub_969240(&v22);
      sub_969240(&v20);
      goto LABEL_16;
    }
    if ( !sub_AAFBB0(a2) )
    {
      sub_AAE940(a1, a2, a2 + 16);
      goto LABEL_16;
    }
    sub_AAE940((__int64)&v22, a2, (__int64)&v18);
    sub_AAE940((__int64)&v25, (__int64)&v18, a2 + 16);
LABEL_29:
    sub_AB3510(a1, (__int64)&v22, (__int64)&v25, 0);
    sub_969240(v27);
    sub_969240(&v25);
    sub_969240(&v24);
    sub_969240(&v22);
    goto LABEL_16;
  }
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 > 0x40 )
  {
    if ( v6 == (unsigned int)sub_C444A0(a2) )
      goto LABEL_8;
LABEL_23:
    v12 = *(_DWORD *)(a2 + 24);
    v13 = a2 + 16;
    if ( v12 <= 0x40 )
    {
      v15 = *(_QWORD *)(a2 + 16);
    }
    else
    {
      v14 = sub_C444A0(a2 + 16);
      v13 = a2 + 16;
      if ( v12 - v14 > 0x40 )
      {
LABEL_27:
        v17 = v13;
        sub_AAE940((__int64)&v22, a2, (__int64)&v18);
        sub_9691E0((__int64)&v20, v5, 1, 0, 0);
        sub_AAE940((__int64)&v25, (__int64)&v20, v17);
        sub_969240(&v20);
        goto LABEL_29;
      }
      v15 = **(_QWORD **)(a2 + 16);
    }
    if ( v15 == 1 )
    {
      sub_AAE940(a1, a2, (__int64)&v18);
      goto LABEL_16;
    }
    goto LABEL_27;
  }
  if ( *(_QWORD *)a2 )
    goto LABEL_23;
LABEL_8:
  v7 = *(_DWORD *)(a2 + 24);
  v8 = a2 + 16;
  if ( v7 <= 0x40 )
  {
    v10 = *(_QWORD *)(a2 + 16);
LABEL_11:
    if ( v10 == 1 )
    {
      sub_AADB10(a1, v6, 0);
      goto LABEL_16;
    }
    goto LABEL_12;
  }
  v9 = sub_C444A0(a2 + 16);
  v8 = a2 + 16;
  if ( v7 - v9 <= 0x40 )
  {
    v10 = **(_QWORD **)(a2 + 16);
    goto LABEL_11;
  }
LABEL_12:
  v16 = v8;
  sub_9691E0((__int64)&v25, v5, 1, 0, 0);
  sub_AAE940(a1, (__int64)&v25, v16);
  sub_969240(&v25);
LABEL_16:
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}
