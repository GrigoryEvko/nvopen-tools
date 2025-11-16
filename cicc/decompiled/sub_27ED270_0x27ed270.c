// Function: sub_27ED270
// Address: 0x27ed270
//
__int64 __fastcall sub_27ED270(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4, _BYTE *a5, unsigned int a6)
{
  unsigned int v6; // r15d
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rsi
  unsigned int v16; // r9d
  _QWORD *v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 *v22; // rax
  _QWORD v26[3]; // [rsp+10h] [rbp-2E0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-2C8h]
  __int64 v28; // [rsp+30h] [rbp-2C0h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-2B8h]
  _QWORD v30[2]; // [rsp+170h] [rbp-180h] BYREF
  char v31; // [rsp+180h] [rbp-170h]
  _BYTE *v32; // [rsp+188h] [rbp-168h]
  __int64 v33; // [rsp+190h] [rbp-160h]
  _BYTE v34[128]; // [rsp+198h] [rbp-158h] BYREF
  __int16 v35; // [rsp+218h] [rbp-D8h]
  _QWORD v36[2]; // [rsp+220h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+230h] [rbp-C0h]
  __int64 v38; // [rsp+238h] [rbp-B8h] BYREF
  unsigned int v39; // [rsp+240h] [rbp-B0h]
  char v40; // [rsp+2B8h] [rbp-38h] BYREF

  v6 = (unsigned __int8)a5[16];
  if ( !(_BYTE)v6 )
  {
    v10 = *a1;
    v26[2] = 0;
    v27 = 1;
    v26[0] = v10;
    v26[1] = v10;
    v11 = &v28;
    do
    {
      *v11 = -4;
      v11 += 5;
      *(v11 - 4) = -3;
      *(v11 - 3) = -4;
      *(v11 - 2) = -3;
    }
    while ( v11 != v30 );
    v30[1] = 0;
    v33 = 0x400000000LL;
    v35 = 256;
    v30[0] = v36;
    v31 = 0;
    v32 = v34;
    v36[1] = 0;
    v37 = 1;
    v36[0] = &unk_49DDBE8;
    v12 = &v38;
    do
    {
      *v12 = -4096;
      v12 += 2;
    }
    while ( v12 != (__int64 *)&v40 );
    v13 = sub_27EB760(a1, (__int64)v26, (__int64)a5, a2);
    v14 = v13;
    if ( v13 == a1[16] )
      goto LABEL_21;
    v15 = *(_QWORD *)(v13 + 64);
    v16 = a6;
    if ( *(_BYTE *)(a3 + 84) )
    {
      v17 = *(_QWORD **)(a3 + 64);
      v18 = &v17[*(unsigned int *)(a3 + 76)];
      if ( v17 == v18 )
      {
LABEL_21:
        v36[0] = &unk_49DDBE8;
        if ( (v37 & 1) == 0 )
          sub_C7D6A0(v38, 16LL * v39, 8);
        nullsub_184();
        if ( v32 != v34 )
          _libc_free((unsigned __int64)v32);
        if ( (v27 & 1) == 0 )
          sub_C7D6A0(v28, 40LL * v29, 8);
        return v6;
      }
      while ( v15 != *v17 )
      {
        if ( v18 == ++v17 )
          goto LABEL_21;
      }
    }
    else
    {
      v22 = sub_C8CA60(a3 + 56, v15);
      v16 = a6;
      if ( !v22 )
        goto LABEL_21;
    }
    v6 = 1;
    if ( (_BYTE)v16 )
    {
      v6 = v16;
      if ( *(_QWORD *)(v14 + 64) == **(_QWORD **)(a3 + 32) )
        LOBYTE(v6) = *(_BYTE *)v14 != 28;
    }
    goto LABEL_21;
  }
  if ( *a5 )
    return v6;
  v7 = *(__int64 **)(a3 + 40);
  v8 = *(__int64 **)(a3 + 32);
  if ( v8 != v7 )
  {
    while ( !(unsigned __int8)sub_27EB670(*v8, (__int64)a1, (__int64)a2) )
    {
      if ( v7 == ++v8 )
        goto LABEL_27;
    }
    return v6;
  }
LABEL_27:
  v19 = *(_QWORD *)(a4 + 40);
  if ( *(_BYTE *)(a3 + 84) )
  {
    v20 = *(_QWORD **)(a3 + 64);
    v21 = &v20[*(unsigned int *)(a3 + 76)];
    if ( v20 == v21 )
      return sub_27EB670(v19, (__int64)a1, (__int64)a2);
    while ( v19 != *v20 )
    {
      if ( v21 == ++v20 )
        return sub_27EB670(v19, (__int64)a1, (__int64)a2);
    }
    return 0;
  }
  if ( sub_C8CA60(a3 + 56, *(_QWORD *)(a4 + 40)) )
    return 0;
  v19 = *(_QWORD *)(a4 + 40);
  return sub_27EB670(v19, (__int64)a1, (__int64)a2);
}
