// Function: sub_185ADB0
// Address: 0x185adb0
//
void __fastcall sub_185ADB0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  _BYTE *v6; // r12
  _BYTE *v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r10
  _QWORD *v13; // r11
  unsigned __int64 v14; // rdi
  __int64 v17; // [rsp+20h] [rbp-70h]
  __int64 v18; // [rsp+28h] [rbp-68h]
  _QWORD *v19; // [rsp+30h] [rbp-60h] BYREF
  char v20; // [rsp+38h] [rbp-58h]
  _BYTE *v21; // [rsp+40h] [rbp-50h] BYREF
  __int64 v22; // [rsp+48h] [rbp-48h]
  _BYTE v23[64]; // [rsp+50h] [rbp-40h] BYREF

  v21 = v23;
  v22 = 0x100000000LL;
  sub_1626700(a1, (__int64)&v21);
  v6 = &v21[8 * (unsigned int)v22];
  if ( v21 != v6 )
  {
    v7 = v21;
    while ( 1 )
    {
      v11 = *(_QWORD *)v7;
      v12 = *(_QWORD *)(*(_QWORD *)v7 - 8LL * *(unsigned int *)(*(_QWORD *)v7 + 8LL));
      v13 = *(_QWORD **)(*(_QWORD *)v7 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v7 + 8LL)));
      if ( a5 > 1 )
      {
        v17 = *(_QWORD *)(*(_QWORD *)v7 - 8LL * *(unsigned int *)(*(_QWORD *)v7 + 8LL));
        v18 = *(_QWORD *)v7;
        sub_15C4EF0((__int64)&v19, v13, a3, a4);
        if ( !v20 )
        {
          v14 = (unsigned __int64)v21;
          if ( v21 == v23 )
            return;
LABEL_12:
          _libc_free(v14);
          return;
        }
        v13 = v19;
        v11 = v18;
        v12 = v17;
      }
      v8 = *(_QWORD *)(v11 + 16);
      v9 = (__int64 *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v8 & 4) != 0 )
        v9 = (__int64 *)*v9;
      v10 = sub_15C5570(v9, v12, (__int64)v13, 0, 1);
      v7 += 8;
      sub_1626A90(a2, v10);
      if ( v6 == v7 )
      {
        v6 = v21;
        break;
      }
    }
  }
  if ( v6 != v23 )
  {
    v14 = (unsigned __int64)v6;
    goto LABEL_12;
  }
}
