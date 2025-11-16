// Function: sub_1816300
// Address: 0x1816300
//
__int64 __fastcall sub_1816300(__int64 a1, __int64 a2)
{
  int v2; // edx
  _BYTE *v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rax
  const void *v7; // r8
  const void *v8; // r9
  signed __int64 v9; // r14
  __int64 v10; // r10
  int v11; // eax
  __int64 v12; // r8
  int v13; // ecx
  __int64 v14; // rdx
  unsigned __int64 v15; // r9
  int v16; // r14d
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r14
  __int64 **v21; // rax
  __int64 *v22; // r12
  __int64 v23; // r12
  _QWORD *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r8
  const void *v28; // [rsp+8h] [rbp-C8h]
  const void *v29; // [rsp+10h] [rbp-C0h]
  __int64 v30; // [rsp+10h] [rbp-C0h]
  int v31; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v32; // [rsp+18h] [rbp-B8h]
  __int64 v33; // [rsp+18h] [rbp-B8h]
  _QWORD *v34; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-A8h]
  _BYTE dest[32]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v37; // [rsp+50h] [rbp-80h]
  __int64 v38; // [rsp+58h] [rbp-78h]
  _QWORD v39[14]; // [rsp+60h] [rbp-70h] BYREF

  v2 = 0;
  v4 = dest;
  v5 = *(unsigned int *)(a2 + 12);
  v6 = *(_QWORD *)(a2 + 16);
  v34 = dest;
  v5 *= 8;
  v7 = (const void *)(v6 + 8);
  v8 = (const void *)(v6 + v5);
  v9 = v5 - 8;
  v35 = 0x400000000LL;
  v10 = v9 >> 3;
  if ( (unsigned __int64)v9 > 0x20 )
  {
    v28 = v8;
    v29 = (const void *)(v6 + 8);
    sub_16CD150((__int64)&v34, dest, v9 >> 3, 8, (int)v7, (int)v8);
    v2 = v35;
    v8 = v28;
    v7 = v29;
    v10 = v9 >> 3;
    v4 = &v34[(unsigned int)v35];
  }
  if ( v7 != v8 )
  {
    v31 = v10;
    memcpy(v4, v7, v9);
    v2 = v35;
    LODWORD(v10) = v31;
  }
  v11 = *(_DWORD *)(a2 + 12);
  v12 = *(_QWORD *)(a1 + 176);
  LODWORD(v35) = v10 + v2;
  v13 = v10 + v2;
  v14 = (unsigned int)(v10 + v2);
  v15 = (unsigned int)(v11 - 1);
  v16 = v11 - 1;
  if ( v15 > (unsigned __int64)HIDWORD(v35) - v14 )
  {
    v30 = v12;
    v32 = (unsigned int)(v11 - 1);
    sub_16CD150((__int64)&v34, dest, v15 + v14, 8, v12, v15);
    v14 = (unsigned int)v35;
    v12 = v30;
    v15 = v32;
    v13 = v35;
  }
  if ( v15 )
  {
    v17 = &v34[v14];
    v18 = &v17[v15];
    do
      *v17++ = v12;
    while ( v18 != v17 );
    v13 = v35;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v20 = (unsigned int)(v13 + v16);
  LODWORD(v35) = v20;
  if ( v19 >> 8 )
  {
    v27 = *(_QWORD *)(a1 + 184);
    if ( (unsigned int)v20 >= HIDWORD(v35) )
    {
      v33 = *(_QWORD *)(a1 + 184);
      sub_16CD150((__int64)&v34, dest, 0, 8, v27, v15);
      v20 = (unsigned int)v35;
      v27 = v33;
    }
    v34[v20] = v27;
    LODWORD(v35) = v35 + 1;
  }
  v21 = *(__int64 ***)(a2 + 16);
  v22 = *v21;
  if ( *((_BYTE *)*v21 + 8) )
  {
    v25 = (_QWORD *)*v22;
    v26 = *(_QWORD *)(a1 + 176);
    v39[0] = v22;
    v37 = v39;
    v39[1] = v26;
    v38 = 0x800000002LL;
    v22 = (__int64 *)sub_1645600(v25, v39, 2, 0);
    if ( v37 != v39 )
      _libc_free((unsigned __int64)v37);
  }
  v23 = sub_1644EA0(v22, v34, (unsigned int)v35, *(_DWORD *)(a2 + 8) >> 8 != 0);
  if ( v34 != (_QWORD *)dest )
    _libc_free((unsigned __int64)v34);
  return v23;
}
