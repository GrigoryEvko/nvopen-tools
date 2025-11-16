// Function: sub_FFF450
// Address: 0xfff450
//
unsigned __int64 __fastcall sub_FFF450(__int64 a1, __int16 a2)
{
  unsigned __int64 result; // rax
  __m128i *v3; // r8
  __int64 *v4; // r13
  int v5; // ebx
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rdx
  unsigned __int64 v13; // [rsp-40h] [rbp-40h]
  unsigned __int64 v14; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v15; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) )
    return **(_QWORD **)a1;
  v3 = (__m128i *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(_DWORD *)(a1 + 16);
  v6 = v4[1];
  if ( *(_BYTE *)(v6 + 8) == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    v15 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43690((__int64)&v14, -1, 1);
      v3 = (__m128i *)(a1 + 24);
    }
    else
    {
      v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v7;
      if ( !v7 )
        v8 = 0;
      v14 = v8;
    }
    v9 = v5 & 4;
    if ( (v5 & 2) != 0 )
      goto LABEL_9;
  }
  else
  {
    v15 = 1;
    v9 = v5 & 4;
    v14 = 1;
    if ( (v5 & 2) != 0 )
    {
LABEL_9:
      if ( v9 )
      {
        v10 = sub_9B3E70(v4, (__int64 *)&v14, a2 & 0x1F8, 0, v3);
        v11 = v10 & 0x3FC;
LABEL_11:
        v12 = v11 & 0x1FB;
        goto LABEL_12;
      }
      v10 = sub_9B3E70(v4, (__int64 *)&v14, a2 & 0x3FC, 0, v3);
      v12 = v10 & 0x3FC;
      goto LABEL_12;
    }
  }
  if ( v9 )
  {
    v10 = sub_9B3E70(v4, (__int64 *)&v14, a2 & 0x1FB, 0, v3);
    v11 = v10;
    goto LABEL_11;
  }
  v10 = sub_9B3E70(v4, (__int64 *)&v14, a2, 0, v3);
  v12 = (unsigned int)v10;
LABEL_12:
  result = v12 | v10 & 0xFFFFFFFF00000000LL;
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
