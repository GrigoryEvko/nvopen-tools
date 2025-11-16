// Function: sub_18B3870
// Address: 0x18b3870
//
__int64 __fastcall sub_18B3870(__int64 a1, __int64 *a2)
{
  char v2; // r8
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  _BYTE *v10; // rsi
  _BYTE *v11; // rsi
  _BYTE *v12; // rdi
  __int64 v13; // r8
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  _BYTE *v15; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v16; // [rsp+18h] [rbp-38h]
  _BYTE *v17; // [rsp+20h] [rbp-30h]

  v2 = sub_1636800(a1, a2);
  result = 0;
  if ( !v2 )
  {
    v15 = 0;
    v16 = 0;
    v4 = sub_16321A0((__int64)a2, (__int64)"llvm.dbg.declare", 16);
    v17 = 0;
    if ( v4 )
    {
LABEL_5:
      v5 = *(_QWORD *)(v4 + 8);
      if ( v5 )
      {
        while ( 1 )
        {
          v6 = sub_1648700(v5);
          v7 = *((_DWORD *)v6 + 5) & 0xFFFFFFF;
          v8 = v6[-3 * v7];
          v9 = v6[3 * (1 - v7)];
          sub_15F20C0(v6);
          if ( *(_QWORD *)(v8 + 8) )
            goto LABEL_4;
          if ( *(_BYTE *)(v8 + 16) > 0x10u )
            break;
          v14 = v8;
          v10 = v16;
          if ( v16 == v17 )
          {
            sub_127D720((__int64)&v15, v16, &v14);
LABEL_4:
            if ( *(_QWORD *)(v9 + 8) )
              goto LABEL_5;
            goto LABEL_12;
          }
          if ( v16 )
          {
            *(_QWORD *)v16 = v8;
            v10 = v16;
          }
          v16 = v10 + 8;
          if ( *(_QWORD *)(v9 + 8) )
            goto LABEL_5;
LABEL_12:
          if ( *(_BYTE *)(v9 + 16) > 0x10u )
            goto LABEL_5;
          v14 = v9;
          v11 = v16;
          if ( v16 == v17 )
          {
            sub_127D720((__int64)&v15, v16, &v14);
            goto LABEL_5;
          }
          if ( v16 )
          {
            *(_QWORD *)v16 = v9;
            v11 = v16;
          }
          v16 = v11 + 8;
          v5 = *(_QWORD *)(v4 + 8);
          if ( !v5 )
            goto LABEL_17;
        }
        v14 = 0;
        sub_1AEB370(v8, 0);
        goto LABEL_4;
      }
LABEL_17:
      sub_15E3D00(v4);
      v12 = v16;
      while ( v15 != v12 )
      {
        v13 = *((_QWORD *)v12 - 1);
        v12 -= 8;
        v16 = v12;
        if ( *(_BYTE *)(v13 + 16) != 3 || (*(_BYTE *)(v13 + 32) & 0xFu) - 7 <= 1 )
        {
          sub_18B2C00(v13);
          v12 = v16;
        }
      }
      if ( v12 )
        j_j___libc_free_0(v12, v17 - v12);
    }
    return 1;
  }
  return result;
}
