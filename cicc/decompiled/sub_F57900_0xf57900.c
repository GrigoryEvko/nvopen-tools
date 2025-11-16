// Function: sub_F57900
// Address: 0xf57900
//
void __fastcall sub_F57900(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v7; // rsi
  int v8; // r15d
  unsigned int v9; // ebx
  char v10; // bl
  __int64 *v11; // rax
  unsigned int v12; // esi
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  unsigned int v18; // [rsp+28h] [rbp-38h]

  v4 = a3;
  v7 = *(_QWORD *)(a4 + 8);
  if ( v7 == *(_QWORD *)(a2 + 8) )
  {
    v12 = 4;
    goto LABEL_19;
  }
  if ( *(_BYTE *)(v7 + 8) == 14 )
  {
    v8 = sub_AE43A0(a1, v7);
    v9 = sub_BCB060(*(_QWORD *)(a2 + 8));
    if ( v9 == v8 )
    {
      sub_ABEA30((__int64)&v15, v4);
      v14 = v9;
      if ( v9 > 0x40 )
        sub_C43690((__int64)&v13, 0, 0);
      else
        v13 = 0;
      v10 = sub_AB1B10((__int64)&v15, (__int64)&v13);
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 )
      {
        if ( v15 )
          j_j___libc_free_0_0(v15);
      }
      if ( !v10 )
      {
        v11 = (__int64 *)sub_BD5C60(a2);
        v12 = 11;
        a3 = sub_B9C770(v11, 0, 0, 0, 1);
LABEL_19:
        sub_B99FD0(a4, v12, a3);
      }
    }
  }
}
