// Function: sub_20145E0
// Address: 0x20145e0
//
__int64 __fastcall sub_20145E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int16 v7; // dx
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __m128i v13; // xmm5
  __m128i v14; // xmm6
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rax
  unsigned __int64 *v25; // rax
  unsigned int v26; // r12d
  _QWORD v28[2]; // [rsp+0h] [rbp-980h] BYREF
  __m128i v29; // [rsp+10h] [rbp-970h]
  __m128i v30; // [rsp+20h] [rbp-960h]
  __m128i v31; // [rsp+30h] [rbp-950h]
  __m128i v32; // [rsp+40h] [rbp-940h]
  __m128i v33; // [rsp+50h] [rbp-930h]
  __m128i v34; // [rsp+60h] [rbp-920h]
  __m128i v35; // [rsp+70h] [rbp-910h]
  __int16 v36; // [rsp+80h] [rbp-900h]
  char v37; // [rsp+82h] [rbp-8FEh]
  int v38; // [rsp+84h] [rbp-8FCh]
  __int64 v39; // [rsp+88h] [rbp-8F8h]
  __int64 v40; // [rsp+90h] [rbp-8F0h]
  __int64 v41; // [rsp+98h] [rbp-8E8h] BYREF
  __int64 v42; // [rsp+158h] [rbp-828h] BYREF
  __int64 v43; // [rsp+160h] [rbp-820h]
  __int64 v44; // [rsp+168h] [rbp-818h] BYREF
  __int64 v45; // [rsp+228h] [rbp-758h] BYREF
  __int64 v46; // [rsp+230h] [rbp-750h]
  __int64 v47; // [rsp+238h] [rbp-748h] BYREF
  __int64 v48; // [rsp+278h] [rbp-708h] BYREF
  __int64 v49; // [rsp+280h] [rbp-700h]
  __int64 v50; // [rsp+288h] [rbp-6F8h] BYREF
  __int64 v51; // [rsp+2E8h] [rbp-698h] BYREF
  __int64 v52; // [rsp+2F0h] [rbp-690h]
  __int64 v53; // [rsp+2F8h] [rbp-688h] BYREF
  __int64 v54; // [rsp+338h] [rbp-648h] BYREF
  __int64 v55; // [rsp+340h] [rbp-640h]
  __int64 v56; // [rsp+348h] [rbp-638h] BYREF
  __int64 v57; // [rsp+388h] [rbp-5F8h] BYREF
  __int64 v58; // [rsp+390h] [rbp-5F0h]
  __int64 v59; // [rsp+398h] [rbp-5E8h] BYREF
  __int64 v60; // [rsp+3F8h] [rbp-588h] BYREF
  __int64 v61; // [rsp+400h] [rbp-580h]
  __int64 v62; // [rsp+408h] [rbp-578h] BYREF
  __int64 v63; // [rsp+448h] [rbp-538h] BYREF
  __int64 v64; // [rsp+450h] [rbp-530h]
  __int64 v65; // [rsp+458h] [rbp-528h] BYREF
  __int64 v66; // [rsp+4B8h] [rbp-4C8h] BYREF
  __int64 v67; // [rsp+4C0h] [rbp-4C0h]
  __int64 v68; // [rsp+4C8h] [rbp-4B8h] BYREF
  __int64 v69; // [rsp+508h] [rbp-478h] BYREF
  __int64 v70; // [rsp+510h] [rbp-470h]
  __int64 v71; // [rsp+518h] [rbp-468h] BYREF
  unsigned __int64 v72[2]; // [rsp+558h] [rbp-428h] BYREF
  _BYTE v73[1048]; // [rsp+568h] [rbp-418h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v28[1] = a1;
  v38 = 1;
  v7 = *(_WORD *)(v6 + 74012);
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 73900));
  v28[0] = v6;
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 73916));
  v10 = _mm_loadu_si128((const __m128i *)(v6 + 73932));
  v39 = 0;
  v11 = _mm_loadu_si128((const __m128i *)(v6 + 73948));
  v12 = _mm_loadu_si128((const __m128i *)(v6 + 73964));
  v40 = 1;
  v13 = _mm_loadu_si128((const __m128i *)(v6 + 73980));
  v14 = _mm_loadu_si128((const __m128i *)(v6 + 73996));
  v36 = v7;
  LOBYTE(v6) = *(_BYTE *)(v6 + 74014);
  v29 = v8;
  v30 = v9;
  v37 = v6;
  v15 = &v41;
  v31 = v10;
  v32 = v11;
  v33 = v12;
  v34 = v13;
  v35 = v14;
  do
  {
    *v15 = 0;
    v15 += 3;
    *((_DWORD *)v15 - 4) = -1;
  }
  while ( v15 != &v42 );
  v16 = &v44;
  v42 = 0;
  v43 = 1;
  do
  {
    *(_DWORD *)v16 = -1;
    v16 += 3;
  }
  while ( v16 != &v45 );
  v17 = &v47;
  v45 = 0;
  v46 = 1;
  do
    *(_DWORD *)v17++ = -1;
  while ( v17 != &v48 );
  v18 = &v50;
  v48 = 0;
  v49 = 1;
  do
  {
    *(_DWORD *)v18 = -1;
    v18 = (__int64 *)((char *)v18 + 12);
  }
  while ( v18 != &v51 );
  v19 = &v53;
  v51 = 0;
  v52 = 1;
  do
    *(_DWORD *)v19++ = -1;
  while ( v19 != &v54 );
  v20 = &v56;
  v54 = 0;
  v55 = 1;
  do
    *(_DWORD *)v20++ = -1;
  while ( v20 != &v57 );
  v21 = &v59;
  v57 = 0;
  v58 = 1;
  do
  {
    *(_DWORD *)v21 = -1;
    v21 = (__int64 *)((char *)v21 + 12);
  }
  while ( v21 != &v60 );
  v22 = &v62;
  v60 = 0;
  v61 = 1;
  do
    *(_DWORD *)v22++ = -1;
  while ( v22 != &v63 );
  v23 = &v65;
  v63 = 0;
  v64 = 1;
  do
  {
    *(_DWORD *)v23 = -1;
    v23 = (__int64 *)((char *)v23 + 12);
  }
  while ( v23 != &v66 );
  v24 = &v68;
  v66 = 0;
  v67 = 1;
  do
    *(_DWORD *)v24++ = -1;
  while ( v24 != &v69 );
  v25 = (unsigned __int64 *)&v71;
  v69 = 0;
  v70 = 1;
  do
    *(_DWORD *)v25++ = -1;
  while ( v25 != v72 );
  v72[0] = (unsigned __int64)v73;
  v72[1] = 0x8000000000LL;
  v26 = sub_2013DC0((__int64)v28, a2, (__int64)v72, a4, a5, a6);
  if ( (_BYTE *)v72[0] != v73 )
    _libc_free(v72[0]);
  if ( (v70 & 1) == 0 )
    j___libc_free_0(v71);
  if ( (v67 & 1) == 0 )
    j___libc_free_0(v68);
  if ( (v64 & 1) == 0 )
    j___libc_free_0(v65);
  if ( (v61 & 1) == 0 )
    j___libc_free_0(v62);
  if ( (v58 & 1) == 0 )
    j___libc_free_0(v59);
  if ( (v55 & 1) == 0 )
    j___libc_free_0(v56);
  if ( (v52 & 1) == 0 )
    j___libc_free_0(v53);
  if ( (v49 & 1) == 0 )
    j___libc_free_0(v50);
  if ( (v46 & 1) == 0 )
    j___libc_free_0(v47);
  if ( (v43 & 1) == 0 )
    j___libc_free_0(v44);
  if ( (v40 & 1) == 0 )
    j___libc_free_0(v41);
  return v26;
}
