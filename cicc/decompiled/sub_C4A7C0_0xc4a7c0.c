// Function: sub_C4A7C0
// Address: 0xc4a7c0
//
__int64 __fastcall sub_C4A7C0(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v7; // ebx
  unsigned int v8; // edx
  _QWORD *v9; // rsi
  unsigned int v10; // eax
  unsigned int v11; // ebx
  bool v12; // cl
  int v13; // eax
  unsigned int v15; // ebx
  int v16; // eax
  unsigned int v17; // [rsp+8h] [rbp-48h]
  unsigned int v18; // [rsp+Ch] [rbp-44h]
  unsigned int v19; // [rsp+Ch] [rbp-44h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-38h]

  sub_C472A0(a1, a2, (__int64 *)a3);
  v7 = *(_DWORD *)(a3 + 8);
  if ( v7 <= 0x40 )
  {
    if ( *(_QWORD *)a3 )
      goto LABEL_3;
LABEL_11:
    *a4 = 0;
    return a1;
  }
  if ( v7 - (unsigned int)sub_C444A0(a3) <= 0x40 && !**(_QWORD **)a3 )
    goto LABEL_11;
LABEL_3:
  sub_C4A3E0((__int64)&v20, a1, a3);
  v8 = v21;
  if ( v21 <= 0x40 )
  {
    v9 = *(_QWORD **)a2;
    if ( v20 != *(_QWORD *)a2 )
    {
      *a4 = 1;
      return a1;
    }
LABEL_6:
    v10 = *(_DWORD *)(a2 + 8);
    v11 = v10 - 1;
    if ( v10 <= 0x40 )
    {
      v12 = 0;
      if ( (_QWORD *)(1LL << v11) != v9 )
        goto LABEL_9;
    }
    else
    {
      v12 = 0;
      if ( (v9[v11 >> 6] & (1LL << v11)) == 0 )
        goto LABEL_9;
      v17 = v8;
      v13 = sub_C44590(a2);
      v8 = v17;
      v12 = 0;
      if ( v13 != v11 )
        goto LABEL_9;
    }
    v15 = *(_DWORD *)(a3 + 8);
    v12 = 1;
    if ( v15 )
    {
      if ( v15 <= 0x40 )
      {
        v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == *(_QWORD *)a3;
      }
      else
      {
        v19 = v8;
        v16 = sub_C445E0(a3);
        v8 = v19;
        v12 = v15 == v16;
      }
    }
LABEL_9:
    *a4 = v12;
    if ( v8 <= 0x40 )
      return a1;
    goto LABEL_17;
  }
  v18 = v21;
  if ( sub_C43C50((__int64)&v20, (const void **)a2) )
  {
    v9 = *(_QWORD **)a2;
    v8 = v18;
    goto LABEL_6;
  }
  *a4 = 1;
LABEL_17:
  if ( v20 )
    j_j___libc_free_0_0(v20);
  return a1;
}
