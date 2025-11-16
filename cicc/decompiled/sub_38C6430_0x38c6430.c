// Function: sub_38C6430
// Address: 0x38c6430
//
void __fastcall sub_38C6430(__int64 *a1, int a2)
{
  __int64 v2; // rbx
  bool v3; // cc
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 i; // r12
  char v7[8]; // [rsp+18h] [rbp-B8h] BYREF
  void *src; // [rsp+20h] [rbp-B0h]
  __int64 v9; // [rsp+28h] [rbp-A8h]
  unsigned int v10; // [rsp+30h] [rbp-A0h]
  __int64 v11; // [rsp+38h] [rbp-98h]
  __int64 v12; // [rsp+40h] [rbp-90h]
  char v13; // [rsp+48h] [rbp-88h]
  char v14; // [rsp+50h] [rbp-80h]
  __int64 v15; // [rsp+60h] [rbp-70h] BYREF
  __int64 v16; // [rsp+68h] [rbp-68h] BYREF
  void *v17; // [rsp+70h] [rbp-60h]
  __int64 v18; // [rsp+78h] [rbp-58h]
  __int64 v19; // [rsp+80h] [rbp-50h]
  __int64 v20; // [rsp+88h] [rbp-48h]
  __int64 v21; // [rsp+90h] [rbp-40h]
  char v22; // [rsp+98h] [rbp-38h]
  char v23; // [rsp+A0h] [rbp-30h]
  char v24; // [rsp+A8h] [rbp-28h]

  v2 = a1[1];
  if ( !*(_QWORD *)(v2 + 1016) )
    return;
  v3 = *(_WORD *)(v2 + 1160) <= 4u;
  v24 = 0;
  if ( !v3 )
  {
    sub_167FAB0((__int64)v7, 4, 1);
    v14 = 0;
    v14 = *(_BYTE *)(*(_QWORD *)(v2 + 16) + 356LL);
    if ( v14 )
    {
      v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 32) + 96LL) + 8LL);
      if ( v24 )
        goto LABEL_11;
    }
    else
    {
      v15 = 0;
      if ( v24 )
      {
LABEL_11:
        j___libc_free_0((unsigned __int64)v17);
        LODWORD(v19) = v10;
        if ( v10 )
        {
          v17 = (void *)sub_22077B0(24LL * v10);
          v18 = v9;
          memcpy(v17, src, 24LL * (unsigned int)v19);
        }
        else
        {
          v17 = 0;
          v18 = 0;
        }
        v20 = v11;
        v21 = v12;
        v22 = v13;
        v23 = v14;
LABEL_14:
        sub_167FA50((__int64)v7);
        goto LABEL_3;
      }
    }
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    j___libc_free_0(0);
    LODWORD(v19) = v10;
    if ( v10 )
    {
      v17 = (void *)sub_22077B0(24LL * v10);
      v18 = v9;
      memcpy(v17, src, 24LL * (unsigned int)v19);
    }
    else
    {
      v17 = 0;
      v18 = 0;
    }
    v24 = 1;
    v20 = v11;
    v21 = v12;
    v22 = v13;
    v23 = v14;
    goto LABEL_14;
  }
LABEL_3:
  v4 = *(_QWORD *)(v2 + 32);
  v5 = v2 + 984;
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(v4 + 88), 0);
  for ( i = *(_QWORD *)(v5 + 16); v5 != i; i = sub_220EF30(i) )
    sub_38C6100((__int64 *)(i + 40), a1, (unsigned __int16)a2 | (BYTE2(a2) << 16), (__int64)&v15);
  if ( v24 )
  {
    sub_38C5570((__int64)&v15, a1);
    if ( v24 )
      sub_167FA50((__int64)&v16);
  }
}
