// Function: sub_1758BB0
// Address: 0x1758bb0
//
_QWORD *__fastcall sub_1758BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // rdi
  unsigned __int8 v5; // al
  __int64 v6; // r12
  __int64 *v7; // rbx
  int v8; // eax
  _QWORD *v9; // r13
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD **v13; // rax
  _QWORD *v14; // r14
  __int64 *v15; // rax
  __int64 v16; // rsi
  const void *v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r12
  _QWORD **v21; // rax
  _QWORD *v22; // r14
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  const void *v28; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-68h]
  const void *v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-58h]
  const void *v32; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-48h]
  _BYTE v34[16]; // [rsp+50h] [rbp-40h] BYREF
  __int16 v35; // [rsp+60h] [rbp-30h]

  v4 = *(_BYTE **)(a3 - 48);
  v5 = v4[16];
  v6 = (__int64)(v4 + 24);
  if ( v5 == 13 )
  {
LABEL_2:
    v8 = *(unsigned __int16 *)(a2 + 18);
    v7 = *(__int64 **)(a3 - 24);
    BYTE1(v8) &= ~0x80u;
    if ( v8 == 34 )
    {
      v29 = *(_DWORD *)(a4 + 8);
      if ( v29 > 0x40 )
        sub_16A4FD0((__int64)&v28, (const void **)a4);
      else
        v28 = *(const void **)a4;
      sub_16A7490((__int64)&v28, 1);
      v18 = v29;
      v29 = 0;
      v31 = v18;
      v30 = v28;
      sub_16A9D70((__int64)&v32, v6, (__int64)&v30);
      v19 = sub_15A1070(*v7, (__int64)&v32);
      v35 = 257;
      v20 = v19;
      v9 = sub_1648A60(56, 2u);
      if ( v9 )
      {
        v21 = (_QWORD **)*v7;
        if ( *(_BYTE *)(*v7 + 8) == 16 )
        {
          v22 = v21[4];
          v23 = (__int64 *)sub_1643320(*v21);
          v24 = (__int64)sub_16463B0(v23, (unsigned int)v22);
        }
        else
        {
          v24 = sub_1643320(*v21);
        }
        sub_15FEC10((__int64)v9, v24, 51, 37, (__int64)v7, v20, (__int64)v34, 0);
      }
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
      if ( v29 > 0x40 )
      {
        v17 = v28;
        if ( v28 )
          goto LABEL_11;
      }
    }
    else
    {
      v9 = 0;
      if ( v8 == 36 )
      {
        sub_16A9D70((__int64)&v32, v6, a4);
        v11 = sub_15A1070(*v7, (__int64)&v32);
        v35 = 257;
        v12 = v11;
        v9 = sub_1648A60(56, 2u);
        if ( v9 )
        {
          v13 = (_QWORD **)*v7;
          if ( *(_BYTE *)(*v7 + 8) == 16 )
          {
            v14 = v13[4];
            v15 = (__int64 *)sub_1643320(*v13);
            v16 = (__int64)sub_16463B0(v15, (unsigned int)v14);
          }
          else
          {
            v16 = sub_1643320(*v13);
          }
          sub_15FEC10((__int64)v9, v16, 51, 34, (__int64)v7, v12, (__int64)v34, 0);
        }
        if ( v33 > 0x40 )
        {
          v17 = v32;
          if ( v32 )
          {
LABEL_11:
            j_j___libc_free_0_0(v17);
            return v9;
          }
        }
      }
    }
    return v9;
  }
  if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 16 )
  {
    v26 = a4;
    v27 = a3;
    if ( v5 <= 0x10u )
    {
      v25 = sub_15A1020(v4, a2, a3, a4);
      if ( v25 )
      {
        if ( *(_BYTE *)(v25 + 16) == 13 )
        {
          a3 = v27;
          v6 = v25 + 24;
          a4 = v26;
          goto LABEL_2;
        }
      }
    }
  }
  return 0;
}
