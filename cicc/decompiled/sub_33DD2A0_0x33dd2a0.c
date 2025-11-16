// Function: sub_33DD2A0
// Address: 0x33dd2a0
//
__int64 __fastcall sub_33DD2A0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned int v12; // r15d
  __int64 v13; // rcx
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // [rsp+0h] [rbp-70h]
  unsigned __int8 v19; // [rsp+8h] [rbp-68h]
  __int16 v20; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+18h] [rbp-58h]
  unsigned __int64 v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+30h] [rbp-40h]
  __int64 v25; // [rsp+38h] [rbp-38h]

  v6 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v20 = v7;
  v21 = v8;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      LOWORD(v22) = v7;
      v23 = v8;
      goto LABEL_4;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v10 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v20) )
    {
      v23 = v8;
      LOWORD(v22) = 0;
      goto LABEL_9;
    }
    LOWORD(v7) = sub_3009970((__int64)&v20, a2, v15, v16, v17);
  }
  LOWORD(v22) = v7;
  v23 = v10;
  if ( !(_WORD)v7 )
  {
LABEL_9:
    v9 = sub_3007260((__int64)&v22);
    v24 = v9;
    v25 = v11;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
    BUG();
  v9 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
LABEL_10:
  v12 = v9 - 1;
  LODWORD(v23) = v9;
  v13 = 1LL << ((unsigned __int8)v9 - 1);
  if ( (unsigned int)v9 <= 0x40 )
  {
    v22 = 0;
LABEL_12:
    v22 |= v13;
    goto LABEL_13;
  }
  v18 = 1LL << ((unsigned __int8)v9 - 1);
  sub_C43690((__int64)&v22, 0, 0);
  v13 = v18;
  if ( (unsigned int)v23 <= 0x40 )
    goto LABEL_12;
  *(_QWORD *)(v22 + 8LL * (v12 >> 6)) |= v18;
LABEL_13:
  result = sub_33DD210(a1, a2, a3, (__int64)&v22, a4);
  if ( (unsigned int)v23 > 0x40 )
  {
    if ( v22 )
    {
      v19 = result;
      j_j___libc_free_0_0(v22);
      return v19;
    }
  }
  return result;
}
