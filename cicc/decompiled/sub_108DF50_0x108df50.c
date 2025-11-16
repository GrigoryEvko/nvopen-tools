// Function: sub_108DF50
// Address: 0x108df50
//
__int64 __fastcall sub_108DF50(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int16 a5,
        __int16 a6,
        char a7,
        char a8)
{
  unsigned __int64 v9; // r13
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned int v13; // eax
  __int64 v14; // rdi
  unsigned __int32 v15; // edx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  char *v21; // r12
  __int64 v22; // r14
  unsigned __int8 v23; // al
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rdi
  unsigned __int32 v28; // edx
  __int64 v29; // rdi
  unsigned __int32 v30; // eax
  __int16 v33; // [rsp+2Ch] [rbp-54h]
  __int16 v34; // [rsp+2Eh] [rbp-52h]
  unsigned __int8 v35; // [rsp+3Fh] [rbp-41h] BYREF
  char dest[8]; // [rsp+40h] [rbp-40h] BYREF
  char v37; // [rsp+48h] [rbp-38h] BYREF

  v9 = a4;
  v34 = a5;
  v33 = a6;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
  {
    v11 = *(_QWORD *)(a1 + 168);
    if ( *(_DWORD *)(a1 + 176) != 1 )
      v9 = _byteswap_uint64(a4);
    *(_QWORD *)dest = v9;
    sub_CB6200(v11, (unsigned __int8 *)dest, 8u);
    v12 = sub_C94890(a2, a3);
    v13 = sub_C0C3A0(a1 + 192, a2, (v12 << 32) | (unsigned int)a3);
    v14 = *(_QWORD *)(a1 + 168);
    v15 = v13;
    if ( *(_DWORD *)(a1 + 176) != 1 )
      v15 = _byteswap_ulong(v13);
    *(_DWORD *)dest = v15;
    sub_CB6200(v14, (unsigned __int8 *)dest, 4u);
  }
  else
  {
    if ( a3 > 8 )
    {
      v24 = *(_QWORD *)(a1 + 168);
      *(_DWORD *)dest = 0;
      sub_CB6200(v24, (unsigned __int8 *)dest, 4u);
      v25 = sub_C94890(a2, a3);
      v26 = sub_C0C3A0(a1 + 192, a2, (v25 << 32) | (unsigned int)a3);
      v27 = *(_QWORD *)(a1 + 168);
      v28 = v26;
      if ( *(_DWORD *)(a1 + 176) != 1 )
        v28 = _byteswap_ulong(v26);
      *(_DWORD *)dest = v28;
      sub_CB6200(v27, (unsigned __int8 *)dest, 4u);
    }
    else
    {
      v21 = dest;
      strncpy(dest, a2, 8u);
      v22 = *(_QWORD *)(a1 + 168);
      do
      {
        v23 = *v21++;
        v35 = v23;
        sub_CB6200(v22, &v35, 1u);
      }
      while ( &v37 != v21 );
    }
    v29 = *(_QWORD *)(a1 + 168);
    v30 = v9;
    if ( *(_DWORD *)(a1 + 176) != 1 )
      v30 = _byteswap_ulong(v9);
    *(_DWORD *)dest = v30;
    sub_CB6200(v29, (unsigned __int8 *)dest, 4u);
  }
  v16 = *(_QWORD *)(a1 + 168);
  if ( *(_DWORD *)(a1 + 176) != 1 )
    v34 = __ROL2__(a5, 8);
  *(_WORD *)dest = v34;
  sub_CB6200(v16, (unsigned __int8 *)dest, 2u);
  v17 = *(_QWORD *)(a1 + 168);
  if ( *(_DWORD *)(a1 + 176) != 1 )
    v33 = __ROL2__(a6, 8);
  *(_WORD *)dest = v33;
  sub_CB6200(v17, (unsigned __int8 *)dest, 2u);
  v18 = *(_QWORD *)(a1 + 168);
  dest[0] = a7;
  sub_CB6200(v18, (unsigned __int8 *)dest, 1u);
  v19 = *(_QWORD *)(a1 + 168);
  dest[0] = a8;
  return sub_CB6200(v19, (unsigned __int8 *)dest, 1u);
}
