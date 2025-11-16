// Function: sub_139D140
// Address: 0x139d140
//
__int64 __fastcall sub_139D140(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, char a6, __int64 a7)
{
  char v9; // cl
  unsigned int v10; // r13d
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r8
  char v15; // [rsp+4h] [rbp-6Ch]
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 (__fastcall **v18)(); // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 v21; // [rsp+28h] [rbp-48h]
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]
  unsigned __int8 v24; // [rsp+32h] [rbp-3Eh]

  if ( a5 )
  {
    v9 = a6;
    if ( a7 )
    {
      v19 = a7;
      v18 = off_49847C0;
    }
    else
    {
      v12 = *(_QWORD *)(a4 + 40);
      v13 = sub_22077B0(552);
      v9 = a6;
      if ( v13 )
      {
        v15 = a6;
        v17 = v13;
        sub_143ACA0(v13, v12);
        v22 = a2;
        v18 = off_49847C0;
        v19 = v17;
        v20 = a4;
        v21 = a5;
        v23 = v15;
        v24 = 0;
        sub_139C7C0(a1, (__int64)&v18);
        v14 = v17;
        if ( (*(_BYTE *)(v17 + 8) & 1) == 0 )
        {
          j___libc_free_0(*(_QWORD *)(v17 + 16));
          v14 = v17;
        }
        j_j___libc_free_0(v14, 552);
        goto LABEL_5;
      }
      v19 = 0;
      v18 = off_49847C0;
    }
    v22 = a2;
    v20 = a4;
    v21 = a5;
    v23 = v9;
    v24 = 0;
    sub_139C7C0(a1, (__int64)&v18);
LABEL_5:
    v10 = v24;
    v18 = off_49847C0;
    nullsub_518();
    return v10;
  }
  return sub_139D0F0(a1, a2);
}
