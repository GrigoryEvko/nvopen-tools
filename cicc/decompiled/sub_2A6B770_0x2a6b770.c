// Function: sub_2A6B770
// Address: 0x2a6b770
//
void __fastcall sub_2A6B770(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r13
  unsigned __int8 v4; // al
  unsigned __int8 *v5; // rax
  __int64 v6; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-A8h]
  __int64 v8; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-98h]
  __int16 v10; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v11; // [rsp+38h] [rbp-88h]
  unsigned int v12; // [rsp+40h] [rbp-80h]
  unsigned __int64 v13; // [rsp+48h] [rbp-78h]
  unsigned int v14; // [rsp+50h] [rbp-70h]
  __int64 v15; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+68h] [rbp-58h]
  __int64 v17; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+78h] [rbp-48h]
  char v19; // [rsp+80h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *a1;
  v4 = *(_BYTE *)(v2 + 8);
  if ( v4 == 15 )
  {
    sub_2A6A450(*a1, a2);
  }
  else
  {
    if ( (unsigned int)v4 - 17 <= 1 )
      v4 = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
    if ( v4 == 12 && (sub_B2D8F0((__int64)&v15, a2), v19) )
    {
      v7 = v16;
      if ( v16 > 0x40 )
        sub_C43780((__int64)&v6, (const void **)&v15);
      else
        v6 = v15;
      v9 = v18;
      if ( v18 > 0x40 )
        sub_C43780((__int64)&v8, (const void **)&v17);
      else
        v8 = v17;
      sub_22C06B0((__int64)&v10, (__int64)&v6, 0);
      sub_969240(&v8);
      sub_969240(&v6);
      if ( v19 )
      {
        v19 = 0;
        sub_969240(&v17);
        sub_969240(&v15);
      }
    }
    else if ( (unsigned __int8)sub_B2F0A0(a2, 1) )
    {
      v5 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(a2 + 8), 1);
      v10 = 0;
      sub_2A62A00((__int64)&v10, v5);
    }
    else
    {
      v10 = 6;
    }
    sub_2A689D0(v3, a2, (unsigned __int8 *)&v10, 0x100000000LL);
    if ( (unsigned int)(unsigned __int8)v10 - 4 <= 1 )
    {
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      if ( v12 > 0x40 )
      {
        if ( v11 )
          j_j___libc_free_0_0(v11);
      }
    }
  }
}
