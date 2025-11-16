// Function: sub_9B3EA0
// Address: 0x9b3ea0
//
unsigned __int64 __fastcall sub_9B3EA0(__int64 *a1, char a2, int a3, unsigned int a4, __m128i *a5)
{
  __int64 v7; // rax
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v14; // [rsp+8h] [rbp-48h]
  __m128i *v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  v7 = a1[1];
  if ( *(_BYTE *)(v7 + 8) == 17 )
  {
    v8 = *(_DWORD *)(v7 + 32);
    v17 = v8;
    if ( v8 > 0x40 )
    {
      v15 = a5;
      sub_C43690(&v16, -1, 1);
      a5 = v15;
    }
    else
    {
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
      if ( !v8 )
        v9 = 0;
      v16 = v9;
    }
  }
  else
  {
    v17 = 1;
    v16 = 1;
  }
  if ( (a2 & 2) != 0 )
  {
    if ( (a2 & 4) != 0 )
    {
      v10 = sub_9B3E70(a1, (__int64 *)&v16, a3 & 0x1F8, a4, a5);
      v11 = v10 & 0x3FC;
LABEL_9:
      v12 = v11 & 0x1FB;
      goto LABEL_10;
    }
    v10 = sub_9B3E70(a1, (__int64 *)&v16, a3 & 0x3FC, a4, a5);
    v12 = v10 & 0x3FC;
  }
  else
  {
    if ( (a2 & 4) != 0 )
    {
      v10 = sub_9B3E70(a1, (__int64 *)&v16, a3 & 0x1FB, a4, a5);
      v11 = v10;
      goto LABEL_9;
    }
    v10 = sub_9B3E70(a1, (__int64 *)&v16, a3, a4, a5);
    v12 = (unsigned int)v10;
  }
LABEL_10:
  result = v12 | v10 & 0xFFFFFFFF00000000LL;
  if ( v17 > 0x40 )
  {
    if ( v16 )
    {
      v14 = result;
      j_j___libc_free_0_0(v16);
      return v14;
    }
  }
  return result;
}
