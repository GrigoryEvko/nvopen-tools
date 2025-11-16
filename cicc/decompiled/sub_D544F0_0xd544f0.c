// Function: sub_D544F0
// Address: 0xd544f0
//
__int64 __fastcall sub_D544F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r13
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 *v21; // rbx
  unsigned __int64 v22; // r13
  __int64 v23; // rdi
  __int64 v25[4]; // [rsp+20h] [rbp-210h] BYREF
  __int64 v26[4]; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v27[4]; // [rsp+60h] [rbp-1D0h] BYREF
  _BYTE v28[8]; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v29; // [rsp+88h] [rbp-1A8h]
  char v30; // [rsp+9Ch] [rbp-194h]
  __int64 v31; // [rsp+E0h] [rbp-150h]
  __int64 v32; // [rsp+E8h] [rbp-148h]
  __int64 v33; // [rsp+F0h] [rbp-140h]
  __int64 v34; // [rsp+F8h] [rbp-138h]
  __int64 v35; // [rsp+100h] [rbp-130h]
  unsigned __int64 v36; // [rsp+108h] [rbp-128h]
  __int64 v37; // [rsp+110h] [rbp-120h]
  __int64 v38; // [rsp+118h] [rbp-118h]
  __int64 v39; // [rsp+120h] [rbp-110h]
  __int64 v40; // [rsp+128h] [rbp-108h]
  _BYTE v41[8]; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v42; // [rsp+148h] [rbp-E8h]
  char v43; // [rsp+15Ch] [rbp-D4h]
  __int64 v44; // [rsp+1A0h] [rbp-90h]
  __int64 v45; // [rsp+1A8h] [rbp-88h]
  __int64 v46; // [rsp+1B0h] [rbp-80h]
  __int64 v47; // [rsp+1B8h] [rbp-78h]
  __int64 v48; // [rsp+1C0h] [rbp-70h]
  unsigned __int64 v49; // [rsp+1C8h] [rbp-68h]
  __int64 v50; // [rsp+1D0h] [rbp-60h]
  __int64 v51; // [rsp+1D8h] [rbp-58h]
  __int64 v52; // [rsp+1E8h] [rbp-48h]

  sub_D53210((__int64)v41, a2, a3, a4, a5, a6);
  v6 = 0;
  sub_D53210((__int64)v28, a1, v7, v8, v9, v10);
  while ( 1 )
  {
    v11 = (__int64 *)v49;
    v12 = v46;
    v13 = v36;
    v14 = ((v50 - v51) >> 5) + 16 * (((__int64)(v52 - v49) >> 3) - 1) + ((v48 - v46) >> 5);
    v15 = (v35 - v33) >> 5;
    if ( v14 == v15 + 16 * (((__int64)(v40 - v36) >> 3) - 1) + ((v37 - v38) >> 5) )
    {
      v27[2] = v48;
      v27[3] = v49;
      v26[2] = v39;
      v11 = v26;
      v26[0] = v37;
      v25[1] = v34;
      v26[1] = v38;
      v26[3] = v40;
      v25[0] = v33;
      v25[2] = v35;
      v25[3] = v36;
      v27[0] = v46;
      v27[1] = v47;
      if ( (unsigned __int8)sub_D542B0(v25, v26, v27) )
        break;
    }
    ++v6;
    sub_D53E10((__int64)v28, (__int64)v11, (__int64 *)v15, v14, v12, v13);
  }
  v16 = v31;
  if ( v31 )
  {
    v17 = (__int64 *)v36;
    v18 = v40 + 8;
    if ( v40 + 8 > v36 )
    {
      do
      {
        v19 = *v17++;
        j_j___libc_free_0(v19, 512);
      }
      while ( v18 > (unsigned __int64)v17 );
      v16 = v31;
    }
    v11 = (__int64 *)(8 * v32);
    j_j___libc_free_0(v16, 8 * v32);
  }
  if ( !v30 )
    _libc_free(v29, v11);
  v20 = v44;
  if ( v44 )
  {
    v21 = (__int64 *)v49;
    v22 = v52 + 8;
    if ( v52 + 8 > v49 )
    {
      do
      {
        v23 = *v21++;
        j_j___libc_free_0(v23, 512);
      }
      while ( v22 > (unsigned __int64)v21 );
      v20 = v44;
    }
    v11 = (__int64 *)(8 * v45);
    j_j___libc_free_0(v20, 8 * v45);
  }
  if ( !v43 )
    _libc_free(v42, v11);
  return v6;
}
