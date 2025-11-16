// Function: sub_F8AA70
// Address: 0xf8aa70
//
__int64 __fastcall sub_F8AA70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r15
  unsigned int v6; // r14d
  int v7; // eax
  __int64 v8; // rax
  _BYTE *v9; // rdx
  char v10; // r9
  _BYTE *v11; // rcx
  unsigned int v12; // esi
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  char v19; // al
  _QWORD *v21; // r14
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // [rsp+18h] [rbp-B8h]
  __int64 v27; // [rsp+20h] [rbp-B0h]
  _BYTE *v28; // [rsp+28h] [rbp-A8h]
  unsigned int v29; // [rsp+38h] [rbp-98h]
  _QWORD v30[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v31; // [rsp+60h] [rbp-70h]
  _BYTE v32[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v33; // [rsp+90h] [rbp-40h]

  v3 = sub_F894B0(a1, *(_QWORD *)(a2 + 32));
  v4 = *(_QWORD *)(a2 + 40);
  v28 = (_BYTE *)v3;
  if ( *(_WORD *)(v4 + 24) )
    goto LABEL_7;
  v5 = *(_QWORD *)(v4 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
  {
    v13 = *(_QWORD *)(v5 + 24);
    if ( v13 && (v13 & (v13 - 1)) == 0 )
    {
      _BitScanReverse64(&v13, v13);
      v7 = v6 + (v13 ^ 0x3F) - 64;
      goto LABEL_5;
    }
LABEL_7:
    v18 = sub_F894B0(a1, v4);
    if ( !*(_BYTE *)(a1 + 514) )
    {
LABEL_8:
      v19 = sub_DBE090(*(_QWORD *)a1, *(_QWORD *)(a2 + 40));
      v11 = (_BYTE *)v18;
      v12 = 19;
      v9 = v28;
      v10 = v19;
      return sub_F810E0((__int64 *)a1, v12, v9, v11, 0, v10);
    }
    if ( (unsigned __int8)sub_D9B720(v4, v4, v14, v15, v16, v17) )
    {
      if ( (unsigned __int8)sub_DBE090(*(_QWORD *)a1, v4) )
        goto LABEL_8;
      v26 = a1 + 520;
    }
    else
    {
      v26 = a1 + 520;
      v31 = 257;
      v33 = 257;
      v21 = sub_BD2C40(72, unk_3F10A14);
      if ( v21 )
        sub_B549F0((__int64)v21, v18, (__int64)v32, 0, 0);
      (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
        *(_QWORD *)(a1 + 608),
        v21,
        v30,
        *(_QWORD *)(a1 + 576),
        *(_QWORD *)(a1 + 584));
      v22 = *(_QWORD *)(a1 + 520);
      v23 = v22;
      v27 = v22 + 16LL * *(unsigned int *)(a1 + 528);
      if ( v22 != v27 )
      {
        do
        {
          v24 = *(_QWORD *)(v23 + 8);
          v25 = *(_DWORD *)v23;
          v23 += 16;
          sub_B99FD0((__int64)v21, v25, v24);
        }
        while ( v27 != v23 );
      }
      v18 = (__int64)v21;
      sub_DBE090(*(_QWORD *)a1, v4);
    }
    v30[0] = v18;
    v33 = 257;
    v30[1] = sub_AD64C0(*(_QWORD *)(v18 + 8), 1, 0);
    v18 = sub_B35180(v26, *(_QWORD *)(v18 + 8), 0x16Du, (__int64)v30, 2u, v29, (__int64)v32);
    goto LABEL_8;
  }
  if ( (unsigned int)sub_C44630(v5 + 24) != 1 )
    goto LABEL_7;
  v7 = sub_C444A0(v5 + 24);
LABEL_5:
  v8 = sub_AD64C0(*(_QWORD *)(v5 + 8), v6 - 1 - v7, 0);
  v9 = v28;
  v10 = 1;
  v11 = (_BYTE *)v8;
  v12 = 26;
  return sub_F810E0((__int64 *)a1, v12, v9, v11, 0, v10);
}
