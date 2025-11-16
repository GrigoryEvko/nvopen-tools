// Function: sub_177F600
// Address: 0x177f600
//
unsigned int *__fastcall sub_177F600(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v6; // ebx
  int v7; // eax
  unsigned int *result; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned int v11; // r14d
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rcx
  char v20; // dl
  __int64 **v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rsi
  int v25; // [rsp+18h] [rbp-68h]
  unsigned int *v26; // [rsp+18h] [rbp-68h]
  __int64 *v27; // [rsp+20h] [rbp-60h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h]
  _BYTE v29[80]; // [rsp+30h] [rbp-50h] BYREF

  v4 = (__int64)(a2 + 24);
  if ( a2[16] != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      goto LABEL_7;
    v22 = sub_15A1020(a2, (__int64)a2, a3, a4);
    if ( !v22 || *(_BYTE *)(v22 + 16) != 13 )
      goto LABEL_7;
    v4 = v22 + 24;
  }
  v6 = *(_DWORD *)(v4 + 8);
  if ( v6 <= 0x40 )
  {
    v9 = *(_QWORD *)v4;
    if ( *(_QWORD *)v4 && (v9 & (v9 - 1)) == 0 )
    {
      _BitScanReverse64(&v9, v9);
      v7 = v6 + (v9 ^ 0x3F) - 64;
      return (unsigned int *)sub_15A0680(a1, v6 - 1 - v7, 0);
    }
  }
  else if ( (unsigned int)sub_16A5940(v4) == 1 )
  {
    v7 = sub_16A57B0(v4);
    return (unsigned int *)sub_15A0680(a1, v6 - 1 - v7, 0);
  }
LABEL_7:
  result = 0;
  if ( *(_BYTE *)(a1 + 8) == 16 )
  {
    v27 = (__int64 *)v29;
    v28 = 0x400000000LL;
    v25 = *(_QWORD *)(a1 + 32);
    if ( v25 )
    {
      v10 = 0;
      while ( 1 )
      {
        result = (unsigned int *)sub_15A0A60((__int64)a2, v10);
        if ( !result )
          goto LABEL_29;
        v20 = *((_BYTE *)result + 16);
        if ( v20 == 9 )
        {
          v21 = (__int64 **)a1;
          if ( *(_BYTE *)(a1 + 8) == 16 )
            v21 = **(__int64 ****)(a1 + 16);
          v17 = sub_1599EF0(v21);
          v18 = (unsigned int)v28;
          if ( (unsigned int)v28 < HIDWORD(v28) )
            goto LABEL_18;
        }
        else
        {
          if ( v20 != 13
            && (*(_BYTE *)(*(_QWORD *)result + 8LL) != 16
             || (result = (unsigned int *)sub_15A1020(result, v10, *(_QWORD *)result, v19)) == 0
             || *((_BYTE *)result + 16) != 13) )
          {
LABEL_28:
            result = 0;
            goto LABEL_29;
          }
          v11 = result[8];
          v12 = (unsigned __int64 *)(result + 6);
          if ( v11 > 0x40 )
          {
            if ( (unsigned int)sub_16A5940((__int64)(result + 6)) != 1 )
              goto LABEL_28;
            LODWORD(v13) = sub_16A57B0((__int64)v12);
          }
          else
          {
            v13 = *v12;
            if ( !*v12 || (v13 & (v13 - 1)) != 0 )
              goto LABEL_28;
            _BitScanReverse64(&v13, v13);
            LODWORD(v13) = v11 + (v13 ^ 0x3F) - 64;
          }
          v14 = a1;
          if ( *(_BYTE *)(a1 + 8) == 16 )
            v14 = **(_QWORD **)(a1 + 16);
          v17 = sub_15A0680(v14, v11 - 1 - (unsigned int)v13, 0);
          v18 = (unsigned int)v28;
          if ( (unsigned int)v28 < HIDWORD(v28) )
            goto LABEL_18;
        }
        sub_16CD150((__int64)&v27, v29, 0, 8, v15, v16);
        v18 = (unsigned int)v28;
LABEL_18:
        ++v10;
        v27[v18] = v17;
        LODWORD(v28) = v28 + 1;
        if ( v25 == v10 )
        {
          v23 = v27;
          v24 = (unsigned int)v28;
          goto LABEL_41;
        }
      }
    }
    v23 = (__int64 *)v29;
    v24 = 0;
LABEL_41:
    result = (unsigned int *)sub_15A01B0(v23, v24);
LABEL_29:
    if ( v27 != (__int64 *)v29 )
    {
      v26 = result;
      _libc_free((unsigned __int64)v27);
      return v26;
    }
  }
  return result;
}
