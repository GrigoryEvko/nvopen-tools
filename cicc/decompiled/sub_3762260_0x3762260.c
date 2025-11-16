// Function: sub_3762260
// Address: 0x3762260
//
__int64 __fastcall sub_3762260(__int64 a1)
{
  __int16 *v1; // rsi
  __int16 *v2; // rsi
  __int64 *v3; // rax
  __int64 *v4; // rax
  __int64 *v5; // rax
  __int64 *v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // rax
  __int64 *v9; // rax
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rax
  unsigned __int64 *v14; // rax
  unsigned int v15; // r12d
  _QWORD v17[2]; // [rsp+0h] [rbp-A70h] BYREF
  _BYTE v18[272]; // [rsp+10h] [rbp-A60h] BYREF
  __int16 v19; // [rsp+120h] [rbp-950h]
  int v20; // [rsp+124h] [rbp-94Ch]
  __int64 v21; // [rsp+128h] [rbp-948h]
  __int64 v22; // [rsp+130h] [rbp-940h]
  __int64 v23; // [rsp+138h] [rbp-938h] BYREF
  unsigned int v24; // [rsp+140h] [rbp-930h]
  __int64 v25; // [rsp+1F8h] [rbp-878h] BYREF
  __int64 v26; // [rsp+200h] [rbp-870h]
  __int64 v27; // [rsp+208h] [rbp-868h] BYREF
  unsigned int v28; // [rsp+210h] [rbp-860h]
  __int64 v29; // [rsp+2C8h] [rbp-7A8h] BYREF
  __int64 v30; // [rsp+2D0h] [rbp-7A0h]
  __int64 v31; // [rsp+2D8h] [rbp-798h] BYREF
  unsigned int v32; // [rsp+2E0h] [rbp-790h]
  __int64 v33; // [rsp+318h] [rbp-758h] BYREF
  __int64 v34; // [rsp+320h] [rbp-750h]
  __int64 v35; // [rsp+328h] [rbp-748h] BYREF
  unsigned int v36; // [rsp+330h] [rbp-740h]
  __int64 v37; // [rsp+388h] [rbp-6E8h] BYREF
  __int64 v38; // [rsp+390h] [rbp-6E0h]
  __int64 v39; // [rsp+398h] [rbp-6D8h] BYREF
  unsigned int v40; // [rsp+3A0h] [rbp-6D0h]
  __int64 v41; // [rsp+3D8h] [rbp-698h] BYREF
  __int64 v42; // [rsp+3E0h] [rbp-690h]
  __int64 v43; // [rsp+3E8h] [rbp-688h] BYREF
  unsigned int v44; // [rsp+3F0h] [rbp-680h]
  __int64 v45; // [rsp+428h] [rbp-648h] BYREF
  __int64 v46; // [rsp+430h] [rbp-640h]
  __int64 v47; // [rsp+438h] [rbp-638h] BYREF
  unsigned int v48; // [rsp+440h] [rbp-630h]
  __int64 v49; // [rsp+478h] [rbp-5F8h] BYREF
  __int64 v50; // [rsp+480h] [rbp-5F0h]
  __int64 v51; // [rsp+488h] [rbp-5E8h] BYREF
  unsigned int v52; // [rsp+490h] [rbp-5E0h]
  __int64 v53; // [rsp+4E8h] [rbp-588h] BYREF
  __int64 v54; // [rsp+4F0h] [rbp-580h]
  __int64 v55; // [rsp+4F8h] [rbp-578h] BYREF
  unsigned int v56; // [rsp+500h] [rbp-570h]
  __int64 v57; // [rsp+538h] [rbp-538h] BYREF
  __int64 v58; // [rsp+540h] [rbp-530h]
  __int64 v59; // [rsp+548h] [rbp-528h] BYREF
  unsigned int v60; // [rsp+550h] [rbp-520h]
  __int64 v61; // [rsp+5A8h] [rbp-4C8h] BYREF
  __int64 v62; // [rsp+5B0h] [rbp-4C0h]
  __int64 v63; // [rsp+5B8h] [rbp-4B8h] BYREF
  unsigned int v64; // [rsp+5C0h] [rbp-4B0h]
  __int64 v65; // [rsp+5F8h] [rbp-478h] BYREF
  __int64 v66; // [rsp+600h] [rbp-470h]
  __int64 v67; // [rsp+608h] [rbp-468h] BYREF
  unsigned int v68; // [rsp+610h] [rbp-460h]
  unsigned __int64 v69[2]; // [rsp+648h] [rbp-428h] BYREF
  _BYTE v70[1048]; // [rsp+658h] [rbp-418h] BYREF

  v1 = *(__int16 **)(a1 + 16);
  v17[1] = a1;
  v20 = 1;
  v17[0] = v1;
  v1 += 262448;
  qmemcpy(v18, v1, sizeof(v18));
  v2 = v1 + 136;
  v19 = *v2;
  v3 = &v23;
  v21 = 0;
  v22 = 1;
  do
  {
    *v3 = 0;
    v3 += 3;
    *((_DWORD *)v3 - 4) = -1;
  }
  while ( v3 != &v25 );
  v4 = &v27;
  v25 = 0;
  v26 = 1;
  do
  {
    *(_DWORD *)v4 = -1;
    v4 += 3;
  }
  while ( v4 != &v29 );
  v5 = &v31;
  v29 = 0;
  v30 = 1;
  do
    *(_DWORD *)v5++ = -1;
  while ( v5 != &v33 );
  v6 = &v35;
  v33 = 0;
  v34 = 1;
  do
  {
    *(_DWORD *)v6 = -1;
    v6 = (__int64 *)((char *)v6 + 12);
  }
  while ( v6 != &v37 );
  v7 = &v39;
  v37 = 0;
  v38 = 1;
  do
    *(_DWORD *)v7++ = -1;
  while ( v7 != &v41 );
  v8 = &v43;
  v41 = 0;
  v42 = 1;
  do
    *(_DWORD *)v8++ = -1;
  while ( v8 != &v45 );
  v9 = &v47;
  v45 = 0;
  v46 = 1;
  do
    *(_DWORD *)v9++ = -1;
  while ( v9 != &v49 );
  v10 = &v51;
  v49 = 0;
  v50 = 1;
  do
  {
    *(_DWORD *)v10 = -1;
    v10 = (__int64 *)((char *)v10 + 12);
  }
  while ( v10 != &v53 );
  v11 = &v55;
  v53 = 0;
  v54 = 1;
  do
    *(_DWORD *)v11++ = -1;
  while ( v11 != &v57 );
  v12 = &v59;
  v57 = 0;
  v58 = 1;
  do
  {
    *(_DWORD *)v12 = -1;
    v12 = (__int64 *)((char *)v12 + 12);
  }
  while ( v12 != &v61 );
  v13 = &v63;
  v61 = 0;
  v62 = 1;
  do
    *(_DWORD *)v13++ = -1;
  while ( v13 != &v65 );
  v14 = (unsigned __int64 *)&v67;
  v65 = 0;
  v66 = 1;
  do
    *(_DWORD *)v14++ = -1;
  while ( v14 != v69 );
  v69[0] = (unsigned __int64)v70;
  v69[1] = 0x8000000000LL;
  v15 = sub_3761A10((__int64)v17, (__int64)v2);
  if ( (_BYTE *)v69[0] != v70 )
    _libc_free(v69[0]);
  if ( (v66 & 1) == 0 )
    sub_C7D6A0(v67, 8LL * v68, 4);
  if ( (v62 & 1) == 0 )
    sub_C7D6A0(v63, 8LL * v64, 4);
  if ( (v58 & 1) == 0 )
    sub_C7D6A0(v59, 12LL * v60, 4);
  if ( (v54 & 1) == 0 )
    sub_C7D6A0(v55, 8LL * v56, 4);
  if ( (v50 & 1) == 0 )
    sub_C7D6A0(v51, 12LL * v52, 4);
  if ( (v46 & 1) == 0 )
    sub_C7D6A0(v47, 8LL * v48, 4);
  if ( (v42 & 1) == 0 )
    sub_C7D6A0(v43, 8LL * v44, 4);
  if ( (v38 & 1) == 0 )
    sub_C7D6A0(v39, 8LL * v40, 4);
  if ( (v34 & 1) == 0 )
    sub_C7D6A0(v35, 12LL * v36, 4);
  if ( (v30 & 1) == 0 )
    sub_C7D6A0(v31, 8LL * v32, 4);
  if ( (v26 & 1) == 0 )
    sub_C7D6A0(v27, 24LL * v28, 8);
  if ( (v22 & 1) == 0 )
    sub_C7D6A0(v23, 24LL * v24, 8);
  return v15;
}
