// Function: sub_169A290
// Address: 0x169a290
//
__int64 __fastcall sub_169A290(__int64 a1, __int64 a2, char a3, unsigned int a4)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  char v9; // al
  __int64 result; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r14
  unsigned int v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-38h]

  v15 = *(_DWORD *)(a2 + 8);
  v6 = v15;
  v7 = ((unsigned __int64)v15 + 63) >> 6;
  if ( v15 <= 0x40 )
  {
    v8 = *(_QWORD *)a2;
    v9 = *(_BYTE *)(a1 + 18) & 0xF7;
    v14 = v8;
    *(_BYTE *)(a1 + 18) = v9;
    if ( !a3 )
    {
LABEL_3:
      v8 = (__int64)&v14;
      goto LABEL_4;
    }
    v11 = 1LL << ((unsigned __int8)v6 - 1);
    goto LABEL_9;
  }
  sub_16A4FD0(&v14, a2);
  v6 = v15;
  v9 = *(_BYTE *)(a1 + 18) & 0xF7;
  *(_BYTE *)(a1 + 18) = v9;
  if ( a3 )
  {
    v8 = v14;
    v11 = 1LL << ((unsigned __int8)v6 - 1);
    if ( v6 > 0x40 )
    {
      if ( (*(_QWORD *)(v14 + 8LL * ((v6 - 1) >> 6)) & v11) == 0 )
        goto LABEL_4;
      v17 = v6;
      *(_BYTE *)(a1 + 18) = v9 | 8;
      sub_16A4FD0(&v16, &v14);
      LOBYTE(v6) = v17;
      if ( v17 > 0x40 )
      {
        sub_16A8F40(&v16);
        goto LABEL_12;
      }
      v8 = v16;
LABEL_11:
      v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v8;
LABEL_12:
      sub_16A7400(&v16);
      v6 = v17;
      v17 = 0;
      v12 = v16;
      if ( v15 > 0x40 && v14 )
      {
        j_j___libc_free_0_0(v14);
        v14 = v12;
        v15 = v6;
        if ( v17 > 0x40 && v16 )
        {
          j_j___libc_free_0_0(v16);
          v6 = v15;
        }
      }
      else
      {
        v14 = v16;
        v15 = v6;
      }
      goto LABEL_17;
    }
LABEL_9:
    if ( (v11 & v8) == 0 )
      goto LABEL_3;
    v17 = v6;
    *(_BYTE *)(a1 + 18) = v9 | 8;
    goto LABEL_11;
  }
LABEL_17:
  if ( v6 <= 0x40 )
    goto LABEL_3;
  v8 = v14;
LABEL_4:
  result = sub_169A140((__int16 **)a1, v8, (unsigned int)v7, a4);
  if ( v15 > 0x40 )
  {
    if ( v14 )
    {
      v13 = result;
      j_j___libc_free_0_0(v14);
      return v13;
    }
  }
  return result;
}
