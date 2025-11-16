// Function: sub_D56BF0
// Address: 0xd56bf0
//
__int64 __fastcall sub_D56BF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 result; // rax
  __int64 v16; // rdi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 *v21; // rbx
  unsigned __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 *v25; // rbx
  unsigned __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 *v29; // rbx
  unsigned __int64 v30; // r12
  __int64 v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-318h] BYREF
  _BYTE v33[8]; // [rsp+10h] [rbp-310h] BYREF
  __int64 v34; // [rsp+18h] [rbp-308h]
  char v35; // [rsp+2Ch] [rbp-2F4h]
  __int64 v36; // [rsp+70h] [rbp-2B0h]
  __int64 v37; // [rsp+78h] [rbp-2A8h]
  unsigned __int64 v38; // [rsp+98h] [rbp-288h]
  __int64 v39; // [rsp+B8h] [rbp-268h]
  _BYTE v40[8]; // [rsp+D0h] [rbp-250h] BYREF
  __int64 v41; // [rsp+D8h] [rbp-248h]
  char v42; // [rsp+ECh] [rbp-234h]
  __int64 v43; // [rsp+130h] [rbp-1F0h]
  __int64 v44; // [rsp+138h] [rbp-1E8h]
  unsigned __int64 v45; // [rsp+158h] [rbp-1C8h]
  __int64 v46; // [rsp+178h] [rbp-1A8h]
  _BYTE v47[8]; // [rsp+190h] [rbp-190h] BYREF
  __int64 v48; // [rsp+198h] [rbp-188h]
  char v49; // [rsp+1ACh] [rbp-174h]
  __int64 v50; // [rsp+1F0h] [rbp-130h]
  __int64 v51; // [rsp+1F8h] [rbp-128h]
  unsigned __int64 v52; // [rsp+218h] [rbp-108h]
  __int64 v53; // [rsp+238h] [rbp-E8h]
  _BYTE v54[8]; // [rsp+248h] [rbp-D8h] BYREF
  __int64 v55; // [rsp+250h] [rbp-D0h]
  char v56; // [rsp+264h] [rbp-BCh]
  __int64 v57; // [rsp+2A8h] [rbp-78h]
  __int64 v58; // [rsp+2B0h] [rbp-70h]
  unsigned __int64 v59; // [rsp+2D0h] [rbp-50h]
  __int64 v60; // [rsp+2F0h] [rbp-30h]

  v32 = a2;
  *(_DWORD *)a1 = sub_D52BE0(a2, a3);
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  sub_D53380((__int64)v47, &v32, v3);
  sub_D53210((__int64)v40, (__int64)v54, v4, v5, v6, v7);
  sub_D53210((__int64)v33, (__int64)v47, v8, v9, v10, v11);
  v12 = *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16);
  result = sub_D555F0(a1 + 8, v12, (__int64)v33, (__int64)v40, v13, v14);
  v16 = v36;
  if ( v36 )
  {
    v17 = (__int64 *)v38;
    v18 = v39 + 8;
    if ( v39 + 8 > v38 )
    {
      do
      {
        v19 = *v17++;
        j_j___libc_free_0(v19, 512);
      }
      while ( v18 > (unsigned __int64)v17 );
      v16 = v36;
    }
    v12 = 8 * v37;
    result = j_j___libc_free_0(v16, 8 * v37);
  }
  if ( !v35 )
    result = _libc_free(v34, v12);
  v20 = v43;
  if ( v43 )
  {
    v21 = (__int64 *)v45;
    v22 = v46 + 8;
    if ( v46 + 8 > v45 )
    {
      do
      {
        v23 = *v21++;
        j_j___libc_free_0(v23, 512);
      }
      while ( v22 > (unsigned __int64)v21 );
      v20 = v43;
    }
    v12 = 8 * v44;
    result = j_j___libc_free_0(v20, 8 * v44);
  }
  if ( !v42 )
    result = _libc_free(v41, v12);
  v24 = v57;
  if ( v57 )
  {
    v25 = (__int64 *)v59;
    v26 = v60 + 8;
    if ( v60 + 8 > v59 )
    {
      do
      {
        v27 = *v25++;
        j_j___libc_free_0(v27, 512);
      }
      while ( v26 > (unsigned __int64)v25 );
      v24 = v57;
    }
    v12 = 8 * v58;
    result = j_j___libc_free_0(v24, 8 * v58);
  }
  if ( !v56 )
    result = _libc_free(v55, v12);
  v28 = v50;
  if ( v50 )
  {
    v29 = (__int64 *)v52;
    v30 = v53 + 8;
    if ( v53 + 8 > v52 )
    {
      do
      {
        v31 = *v29++;
        j_j___libc_free_0(v31, 512);
      }
      while ( v30 > (unsigned __int64)v29 );
      v28 = v50;
    }
    v12 = 8 * v51;
    result = j_j___libc_free_0(v28, 8 * v51);
  }
  if ( !v49 )
    return _libc_free(v48, v12);
  return result;
}
