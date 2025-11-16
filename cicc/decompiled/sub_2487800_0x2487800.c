// Function: sub_2487800
// Address: 0x2487800
//
__int64 __fastcall sub_2487800(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r15
  __int64 v3; // r13
  _QWORD *v4; // rax
  __int64 v5; // rbx
  unsigned __int64 v6; // rax
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-78h]
  unsigned __int64 v14; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-68h]
  _QWORD v16[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v17; // [rsp+40h] [rbp-40h]

  v1 = sub_BCB2A0(*(_QWORD **)a1);
  v15 = 1;
  v2 = (_QWORD *)v1;
  v14 = (unsigned __int8)qword_4FE93C8;
  BYTE4(v13) = 0;
  v3 = sub_AD6220(v1, (__int64)&v14);
  v16[1] = 19;
  v17 = 261;
  v16[0] = "__memprof_histogram";
  v4 = sub_BD2C40(88, unk_3F0FAE8);
  v5 = (__int64)v4;
  if ( v4 )
    sub_B30000((__int64)v4, a1, v2, 1, 4, v3, (__int64)v16, 0, 0, v13, 0);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  v6 = *(unsigned int *)(a1 + 284);
  if ( (unsigned int)v6 > 8 || (v12 = 292, !_bittest64(&v12, v6)) )
  {
    v7 = *(_BYTE *)(v5 + 32);
    *(_BYTE *)(v5 + 32) = v7 & 0xF0;
    if ( (v7 & 0x30) != 0 )
      *(_BYTE *)(v5 + 33) |= 0x40u;
    v8 = sub_BAA410(a1, "__memprof_histogram", 0x13u);
    sub_B2F990(v5, v8, v9, v10);
  }
  v16[0] = v5;
  return sub_2A41DC0(a1, v16, 1);
}
