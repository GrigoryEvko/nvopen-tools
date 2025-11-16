// Function: sub_9B4D50
// Address: 0x9b4d50
//
__int64 __fastcall sub_9B4D50(
        __int64 a1,
        unsigned int a2,
        __m128i *a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        char a7,
        char a8)
{
  unsigned int v12; // ecx
  unsigned int v13; // edx
  unsigned int v14; // eax
  unsigned int v15; // r12d
  unsigned int v16; // eax
  int v17; // eax
  unsigned int v18; // eax
  unsigned int v23; // [rsp+4h] [rbp-7Ch]
  unsigned int v26; // [rsp+8h] [rbp-78h]
  unsigned int v27; // [rsp+8h] [rbp-78h]
  unsigned int v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-68h]
  _QWORD *v31; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-48h]
  _QWORD *v35; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-38h]

  if ( !a7 && !a8 )
  {
    sub_9B0110((__int64)&v29, a5, a1, a2, a3);
    if ( v32 > 0x40 )
    {
      if ( (*v31 & 1) != 0 )
        goto LABEL_5;
    }
    else if ( ((unsigned __int8)v31 & 1) != 0 )
    {
LABEL_5:
      a4 = sub_9A6530(a6, a1, a3, a2);
LABEL_6:
      if ( v32 > 0x40 && v31 )
        j_j___libc_free_0_0(v31);
      if ( v30 > 0x40 )
      {
        if ( v29 )
          j_j___libc_free_0_0(v29);
      }
      return a4;
    }
    sub_9B0110((__int64)&v33, a6, a1, a2, a3);
    v12 = v36;
    if ( v36 > 0x40 )
    {
      v13 = v32;
      if ( (*v35 & 1) == 0 )
        goto LABEL_17;
    }
    else
    {
      v13 = v32;
      if ( ((unsigned __int8)v35 & 1) == 0 )
      {
LABEL_17:
        if ( v13 <= 0x40 )
        {
          _RAX = v31;
          v15 = 64;
          __asm { tzcnt   rsi, rax }
          if ( v31 )
            v15 = _RSI;
          if ( v13 <= v15 )
            v15 = v13;
        }
        else
        {
          v26 = v36;
          v14 = sub_C44590(&v31);
          v12 = v26;
          v15 = v14;
        }
        if ( v12 <= 0x40 )
        {
          _RDX = v35;
          v16 = 64;
          __asm { tzcnt   rsi, rdx }
          if ( v35 )
            v16 = _RSI;
          if ( v12 <= v16 )
            v16 = v12;
        }
        else
        {
          v27 = v12;
          v16 = sub_C44590(&v35);
          v12 = v27;
        }
        LOBYTE(a4) = v16 + v15 < a4;
        goto LABEL_22;
      }
    }
    if ( v13 <= 0x40 )
    {
      a4 = 1;
      if ( v31 )
      {
LABEL_22:
        if ( v12 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
        if ( v34 > 0x40 && v33 )
          j_j___libc_free_0_0(v33);
        goto LABEL_6;
      }
    }
    else
    {
      v23 = v13;
      a4 = 1;
      v28 = v36;
      v17 = sub_C444A0(&v31);
      v12 = v28;
      if ( v17 != v23 )
        goto LABEL_22;
    }
    v18 = sub_9A6530(a5, a1, a3, a2);
    v12 = v36;
    a4 = v18;
    goto LABEL_22;
  }
  a4 = sub_9A6530(a5, a1, a3, a2);
  if ( !(_BYTE)a4 )
    return a4;
  return sub_9A6530(a6, a1, a3, a2);
}
