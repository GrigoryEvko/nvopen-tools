// Function: sub_1584900
// Address: 0x1584900
//
__int64 __fastcall sub_1584900(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v5; // rsi
  char v6; // al
  __int64 v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  _BYTE *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-170h]
  __int64 v27; // [rsp+18h] [rbp-158h]
  unsigned int v28; // [rsp+24h] [rbp-14Ch]
  unsigned int v29; // [rsp+28h] [rbp-148h]
  _BYTE *v30; // [rsp+30h] [rbp-140h] BYREF
  __int64 v31; // [rsp+38h] [rbp-138h]
  _BYTE v32[304]; // [rsp+40h] [rbp-130h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
  v27 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
  v6 = a3[16];
  if ( v6 == 9 )
  {
    v25 = sub_16463B0(v27, v5);
    return sub_1599EF0(v25);
  }
  if ( v6 == 5 )
    return 0;
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v30 = v32;
  v29 = v7;
  v31 = 0x2000000000LL;
  if ( !(_DWORD)v5 )
  {
    v21 = v32;
    v22 = 0;
    goto LABEL_18;
  }
  v28 = 2 * v7;
  v8 = 0;
  do
  {
    while ( 1 )
    {
      v14 = sub_15FA9D0(a3, v8);
      v15 = v14;
      if ( v14 == -1 )
        break;
      if ( v28 <= v14 )
      {
        v12 = sub_1599EF0(v27);
LABEL_6:
        v13 = (unsigned int)v31;
        if ( (unsigned int)v31 >= HIDWORD(v31) )
          goto LABEL_12;
        goto LABEL_7;
      }
      if ( v14 >= v29 )
      {
        v9 = sub_16498A0(a2);
        v10 = sub_1644900(v9, 32);
        v11 = sub_15A0680(v10, v15 - v29, 0);
        v12 = sub_15A37D0(a2, v11, 0);
        goto LABEL_6;
      }
      v16 = sub_16498A0(a1);
      v17 = sub_1644900(v16, 32);
      v18 = sub_15A0680(v17, (int)v15, 0);
      v12 = sub_15A37D0(a1, v18, 0);
      v13 = (unsigned int)v31;
      if ( (unsigned int)v31 >= HIDWORD(v31) )
      {
LABEL_12:
        v26 = v12;
        sub_16CD150(&v30, v32, 0, 8);
        v13 = (unsigned int)v31;
        v12 = v26;
      }
LABEL_7:
      ++v8;
      *(_QWORD *)&v30[8 * v13] = v12;
      LODWORD(v31) = v31 + 1;
      if ( (_DWORD)v5 == v8 )
        goto LABEL_17;
    }
    v19 = sub_1599EF0(v27);
    v20 = (unsigned int)v31;
    if ( (unsigned int)v31 >= HIDWORD(v31) )
    {
      sub_16CD150(&v30, v32, 0, 8);
      v20 = (unsigned int)v31;
    }
    ++v8;
    *(_QWORD *)&v30[8 * v20] = v19;
    LODWORD(v31) = v31 + 1;
  }
  while ( (_DWORD)v5 != v8 );
LABEL_17:
  v21 = v30;
  v22 = (unsigned int)v31;
LABEL_18:
  v23 = sub_15A01B0(v21, v22);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v23;
}
