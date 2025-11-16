// Function: sub_1F43D80
// Address: 0x1f43d80
//
unsigned __int64 __fastcall sub_1F43D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  char v6; // al
  __int64 v7; // r13
  unsigned int v8; // edx
  unsigned __int8 v9; // r15
  __int64 v10; // r8
  __int64 v11; // rcx
  unsigned int v12; // r14d
  char v14; // al
  __int64 v15; // r9
  unsigned int v16; // edx
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned int v22; // r14d
  unsigned __int8 v23; // al
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  char v33[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]

  v6 = *(_BYTE *)(a3 + 8);
  v7 = *(_QWORD *)a3;
  if ( v6 == 15 )
  {
    v8 = 8 * sub_15A9520(a2, *(_DWORD *)(a3 + 8) >> 8);
    if ( v8 == 32 )
    {
      v9 = 5;
    }
    else if ( v8 > 0x20 )
    {
      v9 = 6;
      if ( v8 != 64 )
      {
        v14 = 7;
        if ( v8 != 128 )
          v14 = 0;
        v9 = v14;
      }
    }
    else
    {
      v9 = 3;
      if ( v8 != 8 )
        v9 = 4 * (v8 == 16);
    }
    v10 = 0;
  }
  else if ( v6 == 16 )
  {
    v15 = *(_QWORD *)(a3 + 24);
    if ( *(_BYTE *)(v15 + 8) == 15 )
    {
      v31 = a4;
      v16 = 8 * sub_15A9520(a2, *(_DWORD *)(v15 + 8) >> 8);
      if ( v16 == 32 )
      {
        v17 = 5;
      }
      else if ( v16 > 0x20 )
      {
        v17 = 6;
        if ( v16 != 64 )
        {
          v17 = 0;
          if ( v16 == 128 )
            v17 = 7;
        }
      }
      else
      {
        v17 = 3;
        if ( v16 != 8 )
          v17 = 4 * (v16 == 16);
      }
      v18 = *(_QWORD *)a3;
      v33[0] = v17;
      v34 = 0;
      v19 = sub_1F58E60(v33, v18);
      a4 = v31;
      v15 = v19;
    }
    v29 = a4;
    v32 = *(_QWORD *)(a3 + 32);
    v20 = sub_1F59570(v15, 0);
    v27 = v21;
    v22 = v20;
    v28 = *(_QWORD *)a3;
    v23 = sub_1D15020(v20, v32);
    v10 = 0;
    v24 = v29;
    v9 = v23;
    if ( !v23 )
    {
      v24 = sub_1F593D0(v28, v22, v27, (unsigned int)v32);
      v9 = v24;
      v10 = v26;
    }
    LOBYTE(v24) = v9;
    v4 = v24;
  }
  else
  {
    v4 = sub_1F59570(a3, 0);
    v9 = v4;
    v10 = v25;
  }
  LOBYTE(v4) = v9;
  v11 = v4;
  v12 = 1;
  while ( 1 )
  {
    LOBYTE(v11) = v9;
    v30 = v10;
    sub_1F40D10((__int64)v33, a1, v7, v11, v10);
    if ( !v33[0] )
      break;
    if ( (v33[0] & 0xFB) == 2 )
      v12 *= 2;
    if ( v9 == (_BYTE)v34 && ((_BYTE)v34 || v30 == v35) )
      return ((unsigned __int64)(unsigned __int8)v34 << 32) | v12;
    v11 = v34;
    v10 = v35;
    v9 = v34;
  }
  return ((unsigned __int64)v9 << 32) | v12;
}
