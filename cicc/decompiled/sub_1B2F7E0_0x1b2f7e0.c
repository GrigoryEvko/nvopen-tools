// Function: sub_1B2F7E0
// Address: 0x1b2f7e0
//
void __fastcall sub_1B2F7E0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // rsi
  char v17; // [rsp+16h] [rbp-12Ah]
  char v18; // [rsp+17h] [rbp-129h]
  __int64 *v19; // [rsp+20h] [rbp-120h]
  __int64 *v20; // [rsp+28h] [rbp-118h]
  __int64 *v21; // [rsp+30h] [rbp-110h]
  __int64 v22; // [rsp+38h] [rbp-108h] BYREF
  __int64 v23; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int64 v24[2]; // [rsp+50h] [rbp-F0h] BYREF
  _QWORD v25[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 *v26; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+78h] [rbp-C8h]
  __int64 v28; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+88h] [rbp-B8h] BYREF
  __int64 *v30[6]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 *v31; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v32; // [rsp+C8h] [rbp-78h]
  _BYTE v33[112]; // [rsp+D0h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 - 24);
  v7 = *(_QWORD *)(a2 - 48);
  v24[0] = (unsigned __int64)v25;
  v8 = *(_QWORD *)(a2 - 72);
  v25[0] = v6;
  v23 = v6;
  v24[1] = 0x200000002LL;
  v26 = &v28;
  v27 = 0x200000000LL;
  v30[1] = &v22;
  v30[2] = &v23;
  v31 = (__int64 *)v33;
  v25[1] = v7;
  v30[3] = a4;
  v30[4] = a1;
  v32 = 0x800000000LL;
  v9 = *(_BYTE *)(v8 + 16);
  v22 = a3;
  v10 = (__int64)v24;
  v30[0] = (__int64 *)v24;
  switch ( v9 )
  {
    case 50:
      v11 = *(_QWORD *)(v8 - 48);
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 75) > 1u
        || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 - 24) + 16LL) - 75) > 1u )
      {
        goto LABEL_5;
      }
      v17 = 0;
      v18 = 1;
      break;
    case 5:
      if ( (*(_WORD *)(v8 + 18) != 26
         || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)) + 16LL) - 75) > 1u
         || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF))) + 16LL) - 75) > 1u)
        && (*(_WORD *)(v8 + 18) != 27
         || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)) + 16LL) - 75) > 1u
         || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF))) + 16LL) - 75) > 1u) )
      {
        goto LABEL_5;
      }
      v18 = 0;
      v11 = *(_QWORD *)(v8 - 48);
      v17 = 0;
      break;
    case 51:
      v11 = *(_QWORD *)(v8 - 48);
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 75) > 1u
        || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v8 - 24) + 16LL) - 75) > 1u )
      {
        goto LABEL_5;
      }
      v18 = 0;
      v17 = 1;
      break;
    default:
      if ( (unsigned __int8)(v9 - 75) > 1u )
        goto LABEL_5;
      v28 = v8;
      v13 = &v28;
      LODWORD(v27) = 1;
      v19 = &v29;
      v17 = 0;
      v18 = 0;
      goto LABEL_14;
  }
  v28 = v11;
  LODWORD(v27) = 1;
  v12 = *(_QWORD *)(v8 - 24);
  LODWORD(v27) = 2;
  v29 = v12;
  sub_16CD150((__int64)&v26, &v28, 0, 8, v7, a6);
  v10 = (unsigned int)v27;
  v26[(unsigned int)v27] = v8;
  v13 = v26;
  LODWORD(v27) = v27 + 1;
  v19 = &v26[(unsigned int)v27];
  if ( v26 != v19 )
  {
LABEL_14:
    v21 = v13;
    do
    {
      v14 = *v21;
      if ( (unsigned __int8)(*(_BYTE *)(*v21 + 16) - 75) > 1u )
      {
        sub_1B2F370(v30, *v21, 0, 0, *v21);
      }
      else
      {
        sub_1B2B720(*v21, (__int64)&v31, v10, (__int64)a4, v7, a6);
        v10 = (__int64)&v31[(unsigned int)v32];
        v20 = (__int64 *)v10;
        if ( v31 != (__int64 *)v10 )
        {
          v15 = v31;
          do
          {
            v16 = *v15++;
            sub_1B2F370(v30, v16, v18, v17, v14);
          }
          while ( v20 != v15 );
        }
      }
      ++v21;
      LODWORD(v32) = 0;
    }
    while ( v19 != v21 );
  }
  if ( v31 != (__int64 *)v33 )
    _libc_free((unsigned __int64)v31);
LABEL_5:
  if ( v26 != &v28 )
    _libc_free((unsigned __int64)v26);
  if ( (_QWORD *)v24[0] != v25 )
    _libc_free(v24[0]);
}
