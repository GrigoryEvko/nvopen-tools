// Function: sub_2175890
// Address: 0x2175890
//
__int64 __fastcall sub_2175890(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned int v8; // eax
  char v9; // di
  int v10; // eax
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rcx
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // r14
  __int64 v22; // [rsp+0h] [rbp-90h] BYREF
  int v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+10h] [rbp-80h] BYREF
  __int64 v25; // [rsp+18h] [rbp-78h]
  __int128 v26; // [rsp+20h] [rbp-70h]
  __int64 v27; // [rsp+30h] [rbp-60h]
  _QWORD v28[10]; // [rsp+40h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a1 + 72);
  v22 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v22, v5, 2);
  v6 = a3[4];
  v23 = *(_DWORD *)(a1 + 64);
  v7 = sub_1E0A0C0(v6);
  v8 = 8 * sub_15A9520(v7, 0);
  if ( v8 == 32 )
  {
    v9 = 5;
    goto LABEL_7;
  }
  if ( v8 > 0x20 )
  {
    if ( v8 == 64 )
    {
      v9 = 6;
      goto LABEL_7;
    }
    if ( v8 == 128 )
    {
      v9 = 7;
      goto LABEL_7;
    }
  }
  else
  {
    if ( v8 == 8 )
    {
      v9 = 3;
      goto LABEL_7;
    }
    v9 = 4;
    if ( v8 == 16 )
    {
LABEL_7:
      LOBYTE(v24) = v9;
      v25 = 0;
      v10 = sub_216FFF0(v9);
      goto LABEL_8;
    }
  }
  v24 = 0;
  v25 = 0;
  v10 = sub_1F58D40((__int64)&v24);
LABEL_8:
  v12 = sub_1D2CC30(a3, (v10 != 32) + 4453, (__int64)&v22, (unsigned int)v24, v25, v11);
  memset(v28, 0, 24);
  v13 = v12;
  v14 = *(__int64 **)(a1 + 32);
  v15 = v14[10];
  LOBYTE(v27) = 0;
  v16 = *(__int64 **)(v15 + 88);
  v17 = 0;
  v26 = (unsigned __int64)v16;
  if ( v16 )
  {
    v18 = *v16;
    if ( *(_BYTE *)(*v16 + 8) == 16 )
      v18 = **(_QWORD **)(v18 + 16);
    v17 = *(_DWORD *)(v18 + 8) >> 8;
  }
  v19 = *v14;
  HIDWORD(v27) = v17;
  v20 = sub_1D2BF40(a3, v19, v14[1], (__int64)&v22, v13, 0, v14[5], v14[6], v26, v27, 0, 0, (__int64)v28);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v20;
}
