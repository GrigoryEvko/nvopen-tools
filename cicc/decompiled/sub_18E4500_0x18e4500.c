// Function: sub_18E4500
// Address: 0x18e4500
//
__int64 __fastcall sub_18E4500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __m128i a6, __m128i a7)
{
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  int v14; // r8d
  unsigned int v15; // r15d
  __int64 *v16; // r10
  __int64 v17; // r9
  __int64 v18; // r14
  unsigned int v19; // ebx
  unsigned int v20; // eax
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 *v24; // rax
  __int64 v25; // r9
  __int64 *v26; // r10
  __int64 v27; // r14
  _BYTE *v28; // rdi
  int v29; // edx
  _BYTE *v30; // r11
  unsigned __int64 v31; // r14
  size_t v32; // r10
  __int64 *v33; // rdi
  _BYTE *src; // [rsp+0h] [rbp-C0h]
  size_t n; // [rsp+8h] [rbp-B8h]
  __int64 *v36; // [rsp+10h] [rbp-B0h]
  __int64 v37; // [rsp+18h] [rbp-A8h]
  __int64 *v38; // [rsp+20h] [rbp-A0h]
  __int64 v39; // [rsp+28h] [rbp-98h]
  _BYTE *v40; // [rsp+30h] [rbp-90h] BYREF
  __int64 v41; // [rsp+38h] [rbp-88h]
  _BYTE dest[32]; // [rsp+40h] [rbp-80h] BYREF
  __int64 *v43; // [rsp+60h] [rbp-60h] BYREF
  __int64 v44; // [rsp+68h] [rbp-58h]
  _BYTE v45[80]; // [rsp+70h] [rbp-50h] BYREF

  v9 = sub_146F1B0((__int64)a5, a4);
  v10 = sub_14806B0((__int64)a5, v9, a1, 0, 0);
  v11 = sub_1456040(a3);
  v12 = sub_147BE00((__int64)a5, v10, v11);
  v13 = sub_14806B0((__int64)a5, v12, a3, 0, 0);
  v15 = sub_18E4240(v13, a2, a5, a6, a7);
  if ( v15 || *(_WORD *)(v13 + 24) != 7 )
    return v15;
  v16 = *(__int64 **)(v13 + 32);
  v17 = *(_QWORD *)(v13 + 40);
  v39 = *v16;
  if ( v17 != 2 )
  {
    v22 = *(_QWORD *)(v13 + 48);
    v23 = v17;
    v40 = dest;
    v37 = v22;
    v24 = &v16[v23];
    v25 = v23 * 8 - 8;
    v26 = v16 + 1;
    v41 = 0x300000000LL;
    v27 = v25 >> 3;
    if ( (unsigned __int64)v25 > 0x18 )
    {
      n = v25;
      v36 = v26;
      v38 = v24;
      sub_16CD150((__int64)&v40, dest, v25 >> 3, 8, v14, v25);
      v30 = v40;
      v29 = v41;
      v24 = v38;
      v26 = v36;
      v25 = n;
      v28 = &v40[8 * (unsigned int)v41];
    }
    else
    {
      v28 = dest;
      v29 = 0;
      v30 = dest;
    }
    if ( v24 != v26 )
    {
      memcpy(v28, v26, v25);
      v30 = v40;
      v29 = v41;
    }
    LODWORD(v41) = v29 + v27;
    v31 = (unsigned int)(v29 + v27);
    v32 = 8 * v31;
    v43 = (__int64 *)v45;
    v44 = 0x400000000LL;
    if ( v31 > 4 )
    {
      src = v30;
      sub_16CD150((__int64)&v43, v45, v31, 8, v14, (int)&v43);
      v32 = 8 * v31;
      v30 = src;
      v33 = &v43[(unsigned int)v44];
    }
    else
    {
      if ( !v32 )
      {
LABEL_17:
        LODWORD(v44) = v32 + v31;
        v18 = sub_14785F0((__int64)a5, &v43, v37, 0);
        if ( v43 != (__int64 *)v45 )
          _libc_free((unsigned __int64)v43);
        if ( v40 != dest )
          _libc_free((unsigned __int64)v40);
        goto LABEL_5;
      }
      v33 = (__int64 *)v45;
    }
    memcpy(v33, v30, v32);
    LODWORD(v32) = v44;
    goto LABEL_17;
  }
  v18 = v16[1];
LABEL_5:
  v19 = sub_18E4240(v39, a2, a5, a6, a7);
  v20 = sub_18E4240(v18, a2, a5, a6, a7);
  if ( v19 && v20 )
  {
    if ( v19 <= v20 )
    {
      if ( v19 >= v20 )
      {
        if ( v19 == v20 )
          return v19;
      }
      else if ( !(v20 % v19) )
      {
        return v19;
      }
    }
    else if ( !(v19 % v20) )
    {
      return v20;
    }
  }
  return v15;
}
