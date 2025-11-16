// Function: sub_247BDB0
// Address: 0x247bdb0
//
void __fastcall sub_247BDB0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rax
  unsigned __int8 *v5; // r12
  __int64 (__fastcall *v6)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r15
  _QWORD *v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // r14
  unsigned int *v13; // r14
  unsigned int *v14; // rbx
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 *v18; // [rsp+8h] [rbp-148h]
  __int64 *v19; // [rsp+18h] [rbp-138h]
  __int64 v20; // [rsp+20h] [rbp-130h]
  _BYTE *v21; // [rsp+28h] [rbp-128h]
  const char *v22; // [rsp+30h] [rbp-120h] BYREF
  char v23; // [rsp+50h] [rbp-100h]
  char v24; // [rsp+51h] [rbp-FFh]
  _BYTE v25[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+80h] [rbp-D0h]
  unsigned int *v27; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v28; // [rsp+98h] [rbp-B8h]
  char v29; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+C8h] [rbp-88h]
  __int64 v31; // [rsp+D0h] [rbp-80h]
  __int64 v32; // [rsp+E0h] [rbp-70h]
  __int64 v33; // [rsp+E8h] [rbp-68h]
  void *v34; // [rsp+110h] [rbp-40h]

  sub_23D0AB0((__int64)&v27, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v2 = *(__int64 **)(a2 - 8);
    v19 = &v2[4 * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)];
  }
  else
  {
    v19 = (__int64 *)a2;
    v2 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  v3 = 0;
  v20 = 0;
  while ( v19 != v2 )
  {
    while ( 1 )
    {
      v9 = *v2;
      v5 = (unsigned __int8 *)sub_246F3F0((__int64)a1, *v2);
      if ( !*(_DWORD *)(a1[1] + 4) )
        break;
      v21 = (_BYTE *)sub_246EE10((__int64)a1, v9);
      if ( !v3 )
      {
        v3 = (unsigned __int64)v5;
        goto LABEL_14;
      }
LABEL_7:
      v4 = sub_2464970(a1, &v27, (unsigned __int64)v5, *(_QWORD *)(v3 + 8), 0);
      v24 = 1;
      v5 = (unsigned __int8 *)v4;
      v23 = 3;
      v22 = "_msprop";
      v6 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v32 + 16LL);
      if ( v6 == sub_9202E0 )
      {
        if ( *(_BYTE *)v3 > 0x15u || *v5 > 0x15u )
        {
LABEL_29:
          v26 = 257;
          v7 = sub_B504D0(29, v3, (__int64)v5, (__int64)v25, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v33 + 16LL))(
            v33,
            v7,
            &v22,
            v30,
            v31);
          v12 = 4LL * v28;
          if ( v27 != &v27[v12] )
          {
            v18 = v2;
            v13 = &v27[v12];
            v14 = v27;
            do
            {
              v15 = *((_QWORD *)v14 + 1);
              v16 = *v14;
              v14 += 4;
              sub_B99FD0(v7, v16, v15);
            }
            while ( v13 != v14 );
            v2 = v18;
          }
          goto LABEL_13;
        }
        if ( (unsigned __int8)sub_AC47B0(29) )
          v7 = sub_AD5570(29, v3, v5, 0, 0);
        else
          v7 = sub_AABE40(0x1Du, (unsigned __int8 *)v3, v5);
      }
      else
      {
        v7 = v6(v32, 29u, (_BYTE *)v3, v5);
      }
      if ( !v7 )
        goto LABEL_29;
LABEL_13:
      v3 = v7;
LABEL_14:
      if ( *(_DWORD *)(a1[1] + 4) )
      {
        if ( v20 )
        {
          if ( *v21 > 0x15u || !sub_AC30F0((__int64)v21) )
          {
            v26 = 257;
            v8 = sub_2465600((__int64)a1, (__int64)v5, (__int64)&v27, (__int64)v25);
            v26 = 257;
            v20 = sub_B36550(&v27, v8, (__int64)v21, v20, (__int64)v25, 0);
          }
        }
        else
        {
          v20 = (__int64)v21;
        }
      }
      v2 += 4;
      if ( v19 == v2 )
        goto LABEL_24;
    }
    if ( v3 )
    {
      v21 = 0;
      goto LABEL_7;
    }
    v3 = (unsigned __int64)v5;
    v2 += 4;
  }
LABEL_24:
  v10 = sub_2463540(a1, *(_QWORD *)(a2 + 8));
  v11 = sub_2464970(a1, &v27, v3, (__int64)v10, 0);
  sub_246EF60((__int64)a1, a2, v11);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_246F1C0((__int64)a1, a2, v20);
  nullsub_61();
  v34 = &unk_49DA100;
  nullsub_63();
  if ( v27 != (unsigned int *)&v29 )
    _libc_free((unsigned __int64)v27);
}
