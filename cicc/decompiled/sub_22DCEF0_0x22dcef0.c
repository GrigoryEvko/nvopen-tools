// Function: sub_22DCEF0
// Address: 0x22dcef0
//
void __fastcall sub_22DCEF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r13
  _BYTE *v4; // r8
  __int64 *v5; // r15
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  _QWORD *v8; // rdx
  _QWORD *v9; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v10; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v11; // [rsp+28h] [rbp-48h]
  _BYTE *v12; // [rsp+30h] [rbp-40h]

  v2 = *a1;
  v9 = a1;
  v10 = 0;
  v11 = 0;
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = 0;
  sub_22DCD60((__int64)&v10, 0, &v9);
  v4 = v11;
  if ( v11 != v10 )
  {
    while ( 1 )
    {
      v5 = (__int64 *)*((_QWORD *)v4 - 1);
      v11 = v4 - 8;
      sub_22DADD0(v5, a2);
      v6 = (_QWORD *)v5[5];
      v7 = (_QWORD *)v5[6];
      v4 = v11;
      if ( v6 != v7 )
        break;
LABEL_10:
      if ( v10 == v4 )
        goto LABEL_11;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = (_QWORD *)*v6;
        if ( v3 == (*(_QWORD *)*v6 & 0xFFFFFFFFFFFFFFF8LL) )
          break;
LABEL_4:
        if ( v7 == ++v6 )
          goto LABEL_10;
      }
      v9 = (_QWORD *)*v6;
      if ( v12 == v4 )
      {
        sub_22DCD60((__int64)&v10, v4, &v9);
        v4 = v11;
        goto LABEL_4;
      }
      if ( v4 )
      {
        *(_QWORD *)v4 = v8;
        v4 = v11;
      }
      v4 += 8;
      ++v6;
      v11 = v4;
      if ( v7 == v6 )
        goto LABEL_10;
    }
  }
LABEL_11:
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
}
