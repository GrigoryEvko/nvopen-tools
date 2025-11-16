// Function: sub_C36910
// Address: 0xc36910
//
__int64 __fastcall sub_C36910(__int64 a1, __int64 a2, char a3, char a4)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  char v9; // al
  __int64 result; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned int v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  unsigned __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  v19 = *(_DWORD *)(a2 + 8);
  v6 = v19;
  v7 = ((unsigned __int64)v19 + 63) >> 6;
  if ( v19 <= 0x40 )
  {
    v8 = *(_QWORD *)a2;
    v9 = *(_BYTE *)(a1 + 20) & 0xF7;
    v18 = v8;
    *(_BYTE *)(a1 + 20) = v9;
    if ( !a3 )
    {
LABEL_3:
      v8 = (__int64)&v18;
      goto LABEL_4;
    }
    v11 = 1LL << ((unsigned __int8)v6 - 1);
    goto LABEL_9;
  }
  sub_C43780(&v18, a2);
  v6 = v19;
  v9 = *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = v9;
  if ( a3 )
  {
    v8 = v18;
    v11 = 1LL << ((unsigned __int8)v6 - 1);
    if ( v6 > 0x40 )
    {
      if ( (*(_QWORD *)(v18 + 8LL * ((v6 - 1) >> 6)) & v11) == 0 )
        goto LABEL_4;
      v21 = v6;
      *(_BYTE *)(a1 + 20) = v9 | 8;
      sub_C43780(&v20, &v18);
      v6 = v21;
      if ( v21 > 0x40 )
      {
        sub_C43D10(&v20, &v18, v14, v15, v16);
LABEL_14:
        sub_C46250(&v20);
        v6 = v21;
        v21 = 0;
        v13 = v20;
        if ( v19 > 0x40 && v18 )
        {
          j_j___libc_free_0_0(v18);
          v18 = v13;
          v19 = v6;
          if ( v21 > 0x40 && v20 )
          {
            j_j___libc_free_0_0(v20);
            v6 = v19;
          }
        }
        else
        {
          v18 = v20;
          v19 = v6;
        }
        goto LABEL_19;
      }
      v8 = v20;
LABEL_11:
      v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v8;
      if ( !v6 )
        v12 = 0;
      v20 = v12;
      goto LABEL_14;
    }
LABEL_9:
    if ( (v11 & v8) == 0 )
      goto LABEL_3;
    v21 = v6;
    *(_BYTE *)(a1 + 20) = v9 | 8;
    goto LABEL_11;
  }
LABEL_19:
  if ( v6 <= 0x40 )
    goto LABEL_3;
  v8 = v18;
LABEL_4:
  result = sub_C367B0(a1, v8, (unsigned int)v7, a4);
  if ( v19 > 0x40 )
  {
    if ( v18 )
    {
      v17 = result;
      j_j___libc_free_0_0(v18);
      return v17;
    }
  }
  return result;
}
