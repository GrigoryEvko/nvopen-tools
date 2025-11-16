// Function: sub_3268530
// Address: 0x3268530
//
__int64 __fastcall sub_3268530(unsigned __int16 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rsi
  unsigned __int64 **v5; // r12
  __int64 v6; // rsi
  int v7; // ebx
  __int64 v8; // rdx
  unsigned __int64 v9; // r14
  __int64 v10; // rdx
  unsigned int v11; // ebx
  unsigned __int64 v12; // r15
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned __int64 *v18; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v20; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-58h]
  __int16 v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+30h] [rbp-40h]
  __int64 v25; // [rsp+38h] [rbp-38h]

  v3 = *a3;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v19 = *(_DWORD *)(v4 + 32);
  v5 = &v18;
  if ( v19 > 0x40 )
    sub_C43780((__int64)&v18, (const void **)(v4 + 24));
  else
    v18 = *(unsigned __int64 **)(v4 + 24);
  v6 = *(_QWORD *)(v3 + 96);
  v21 = *(_DWORD *)(v6 + 32);
  if ( v21 > 0x40 )
    sub_C43780((__int64)&v20, (const void **)(v6 + 24));
  else
    v20 = *(unsigned __int64 **)(v6 + 24);
  sub_3260590((__int64)&v18, (__int64)&v20, 0);
  v7 = *a1;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
    {
      v23 = 0;
      LOWORD(v7) = word_4456580[v7 - 1];
      v22 = v7;
      if ( !(_WORD)v7 )
        goto LABEL_9;
      goto LABEL_28;
    }
    goto LABEL_7;
  }
  if ( !sub_30070B0((__int64)a1) )
  {
LABEL_7:
    v8 = *((_QWORD *)a1 + 1);
    goto LABEL_8;
  }
  LOWORD(v7) = sub_3009970((__int64)a1, (__int64)&v20, v15, v16, v17);
LABEL_8:
  v22 = v7;
  v23 = v8;
  if ( !(_WORD)v7 )
  {
LABEL_9:
    v24 = sub_3007260((__int64)&v22);
    v9 = v24;
    v25 = v10;
    goto LABEL_10;
  }
LABEL_28:
  if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
    BUG();
  v9 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
LABEL_10:
  v11 = v19;
  if ( v19 > 0x40 )
  {
    if ( v11 - (unsigned int)sub_C444A0((__int64)&v18) > 0x40 || (v12 = (unsigned __int64)v18, *v18 >= v9) )
    {
      if ( v21 > 0x40 )
      {
LABEL_13:
        LODWORD(v5) = 0;
        goto LABEL_14;
      }
      v12 = (unsigned __int64)v18;
      LODWORD(v5) = 0;
    }
    else
    {
      LOBYTE(v14) = sub_C43C50((__int64)&v18, (const void **)&v20);
      LODWORD(v5) = v14;
      if ( v21 > 0x40 )
        goto LABEL_14;
    }
LABEL_21:
    if ( v12 )
      j_j___libc_free_0_0(v12);
    return (unsigned int)v5;
  }
  if ( (unsigned __int64)v18 >= v9 )
  {
    LODWORD(v5) = 0;
    if ( v21 <= 0x40 )
      return (unsigned int)v5;
    goto LABEL_13;
  }
  LOBYTE(v5) = v20 == v18;
  if ( v21 <= 0x40 )
    return (unsigned int)v5;
LABEL_14:
  if ( v20 )
  {
    j_j___libc_free_0_0((unsigned __int64)v20);
    v11 = v19;
  }
  if ( v11 > 0x40 )
  {
    v12 = (unsigned __int64)v18;
    goto LABEL_21;
  }
  return (unsigned int)v5;
}
