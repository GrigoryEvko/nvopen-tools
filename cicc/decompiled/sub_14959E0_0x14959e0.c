// Function: sub_14959E0
// Address: 0x14959e0
//
void __fastcall sub_14959E0(__int64 a1, _QWORD *a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 *v6; // r14
  __int64 *i; // rbx
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // rbx
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  unsigned __int64 v17; // rdi
  __int64 v18; // r12
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-170h]
  _BYTE *v22; // [rsp+10h] [rbp-160h] BYREF
  __int64 v23; // [rsp+18h] [rbp-158h]
  _BYTE v24[64]; // [rsp+20h] [rbp-150h] BYREF
  _QWORD v25[5]; // [rsp+60h] [rbp-110h] BYREF
  char *v26; // [rsp+88h] [rbp-E8h]
  char v27; // [rsp+98h] [rbp-D8h] BYREF
  _QWORD *v28; // [rsp+120h] [rbp-50h]
  unsigned int v29; // [rsp+130h] [rbp-40h]

  v6 = *(__int64 **)(a3 + 16);
  for ( i = *(__int64 **)(a3 + 8); v6 != i; ++i )
  {
    v8 = *i;
    sub_14959E0(a1, a2, v8);
  }
  sub_1263B40(a1, "Loop ");
  sub_15537D0(**(_QWORD **)(a3 + 32), a1, 0);
  sub_1263B40(a1, ": ");
  v22 = v24;
  v23 = 0x800000000LL;
  sub_13F9EC0(a3, (__int64)&v22);
  if ( (_DWORD)v23 != 1 )
    sub_1263B40(a1, "<multiple exits> ");
  if ( (unsigned __int8)sub_1481F90(a2, a3, a4, a5) )
  {
    v9 = sub_1263B40(a1, "backedge-taken count is ");
    v10 = sub_1481F60(a2, a3, a4, a5);
    sub_1456620(v10, v9);
  }
  else
  {
    sub_1263B40(a1, "Unpredictable backedge-taken count. ");
  }
  sub_1263B40(a1, "\nLoop ");
  sub_15537D0(**(_QWORD **)(a3 + 32), a1, 0);
  sub_1263B40(a1, ": ");
  v11 = sub_1474260((__int64)a2, a3);
  if ( sub_14562D0(v11) )
  {
    sub_1263B40(a1, "Unpredictable max backedge-taken count. ");
  }
  else
  {
    v12 = sub_1263B40(a1, "max backedge-taken count is ");
    v13 = sub_1474260((__int64)a2, a3);
    sub_1456620(v13, v12);
    if ( (unsigned __int8)sub_1474320((__int64)a2, a3) )
      sub_1263B40(a1, ", actual taken count either this or zero.");
  }
  sub_1263B40(a1, "\nLoop ");
  sub_15537D0(**(_QWORD **)(a3 + 32), a1, 0);
  sub_1263B40(a1, ": ");
  sub_14585E0((__int64)v25);
  v14 = sub_14959A0(a2, a3, (__int64)v25, a4, a5);
  if ( sub_14562D0(v14) )
  {
    sub_1263B40(a1, "Unpredictable predicated backedge-taken count. ");
  }
  else
  {
    v21 = sub_1263B40(a1, "Predicated backedge-taken count is ");
    sub_1456620(v14, v21);
    sub_1263B40(v21, "\n");
    sub_1263B40(a1, " Predicates:\n");
    sub_1452580((__int64)v25, a1, 4u);
  }
  sub_1263B40(a1, "\n");
  if ( (unsigned __int8)sub_1481F90(a2, a3, a4, a5) )
  {
    sub_1263B40(a1, "Loop ");
    sub_15537D0(**(_QWORD **)(a3 + 32), a1, 0);
    sub_1263B40(a1, ": ");
    v18 = sub_1263B40(a1, "Trip multiple is ");
    v19 = sub_147DF00((__int64)a2, a3);
    v20 = sub_16E7A90(v18, v19);
    sub_1263B40(v20, "\n");
  }
  v25[0] = &unk_49EC708;
  if ( v29 )
  {
    v15 = v28;
    v16 = &v28[7 * v29];
    do
    {
      if ( *v15 != -16 && *v15 != -8 )
      {
        v17 = v15[1];
        if ( (_QWORD *)v17 != v15 + 3 )
          _libc_free(v17);
      }
      v15 += 7;
    }
    while ( v16 != v15 );
  }
  j___libc_free_0(v28);
  if ( v26 != &v27 )
    _libc_free((unsigned __int64)v26);
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
}
