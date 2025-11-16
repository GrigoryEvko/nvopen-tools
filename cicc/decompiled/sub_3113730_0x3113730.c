// Function: sub_3113730
// Address: 0x3113730
//
__int64 *__fastcall sub_3113730(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  _BYTE *v9; // rax
  void **v10; // rsi
  __int64 (__fastcall *v11)(__int64); // rax
  char v12; // al
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  _QWORD *v16; // rdi
  __int64 v17; // r14
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v22; // [rsp+0h] [rbp-320h]
  void *v24; // [rsp+10h] [rbp-310h] BYREF
  unsigned __int64 v25; // [rsp+18h] [rbp-308h]
  _QWORD *v26; // [rsp+20h] [rbp-300h] BYREF
  __int64 v27; // [rsp+28h] [rbp-2F8h]
  _BYTE v28[16]; // [rsp+30h] [rbp-2F0h] BYREF
  _QWORD **v29; // [rsp+40h] [rbp-2E0h] BYREF
  unsigned __int64 v30; // [rsp+48h] [rbp-2D8h]
  __int16 v31; // [rsp+60h] [rbp-2C0h]

  if ( *(_BYTE *)(a2 + 104) )
  {
    while ( 1 )
    {
      v3 = 0;
      v4 = sub_C935B0((_QWORD *)(a2 + 120), byte_3F15413, 6, 0);
      v5 = *(_QWORD *)(a2 + 128);
      if ( v4 < v5 )
      {
        v3 = v5 - v4;
        v5 = v4;
      }
      v29 = (_QWORD **)(*(_QWORD *)(a2 + 120) + v5);
      v30 = v3;
      v6 = sub_C93740((__int64 *)&v29, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      if ( v6 > v30 )
        v6 = v30;
      v7 = v30 - v3 + v6;
      if ( v7 > v30 )
        v7 = v30;
      if ( !v7 )
        goto LABEL_20;
      v8 = *(_QWORD *)(a2 + 128);
      if ( !v8 || (v9 = *(_BYTE **)(a2 + 120), *v9 != 58) )
      {
        if ( !*(_BYTE *)(a2 + 104) )
          break;
        v10 = *(void ***)(a2 + 120);
        sub_CB0A90((__int64)&v29, (__int64)v10, *(_QWORD *)(*(_QWORD *)(a2 + 64) + 16LL) - (_QWORD)v10, 0, 0, 0);
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL);
        if ( v11 == sub_3112860 )
          v12 = *(_BYTE *)(a2 + 136) & 1;
        else
          v12 = v11(a2);
        if ( v12 )
        {
          v10 = (void **)&v29;
          sub_3117DB0(a2 + 48, &v29);
        }
        v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 48LL);
        if ( v13 == sub_3112870 )
        {
          if ( (*(_BYTE *)(a2 + 136) & 2) == 0 )
          {
LABEL_18:
            *a1 = 1;
            sub_CB34B0((__int64)&v29, (__int64)v10);
            return a1;
          }
        }
        else if ( !(unsigned __int8)v13(a2) )
        {
          goto LABEL_18;
        }
        v10 = (void **)&v29;
        sub_311FC60(a2 + 56, &v29);
        goto LABEL_18;
      }
      v17 = v8 - 1;
      v29 = (_QWORD **)(v9 + 1);
      v30 = v8 - 1;
      v18 = sub_C93740((__int64 *)&v29, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      if ( v18 > v30 )
        v18 = v30;
      v19 = v30 - v17 + v18;
      if ( v19 > v30 )
        v19 = v30;
      v25 = v19;
      v24 = v29;
      if ( v19 != 18 )
      {
        if ( v19 != 19 )
          goto LABEL_32;
        goto LABEL_38;
      }
      if ( (unsigned int)sub_C92E90(&v24, (__int64)"outlined_hash_tree", 0x12u) )
      {
        if ( v25 != 19 )
          goto LABEL_32;
LABEL_38:
        if ( (unsigned int)sub_C92E90(&v24, (__int64)"stable_function_map", 0x13u) )
        {
LABEL_32:
          *(_DWORD *)(a2 + 8) = 3;
          v27 = 0;
          v26 = v28;
          v28[0] = 0;
          sub_2240AE0((unsigned __int64 *)(a2 + 16), (unsigned __int64 *)&v26);
          v29 = &v26;
          v31 = 260;
          v20 = sub_22077B0(0x30u);
          if ( v20 )
          {
            *(_DWORD *)(v20 + 8) = 3;
            v22 = v20;
            *(_QWORD *)v20 = &unk_4A32A78;
            sub_CA0F50((__int64 *)(v20 + 16), (void **)&v29);
            v20 = v22;
          }
          v16 = v26;
          *a1 = v20 | 1;
          if ( v16 == (_QWORD *)v28 )
            return a1;
LABEL_25:
          j_j___libc_free_0((unsigned __int64)v16);
          return a1;
        }
        *(_DWORD *)(a2 + 136) |= 2u;
        sub_C7C5C0(a2 + 72);
        if ( !*(_BYTE *)(a2 + 104) )
          break;
      }
      else
      {
        *(_DWORD *)(a2 + 136) |= 1u;
LABEL_20:
        sub_C7C5C0(a2 + 72);
        if ( !*(_BYTE *)(a2 + 104) )
          break;
      }
    }
  }
  if ( !*(_DWORD *)(a2 + 136) )
  {
    *a1 = 1;
    return a1;
  }
  *(_DWORD *)(a2 + 8) = 3;
  v27 = 0;
  v26 = v28;
  v28[0] = 0;
  sub_2240AE0((unsigned __int64 *)(a2 + 16), (unsigned __int64 *)&v26);
  v29 = &v26;
  v31 = 260;
  v14 = sub_22077B0(0x30u);
  v15 = v14;
  if ( v14 )
  {
    *(_DWORD *)(v14 + 8) = 3;
    *(_QWORD *)v14 = &unk_4A32A78;
    sub_CA0F50((__int64 *)(v14 + 16), (void **)&v29);
  }
  v16 = v26;
  *a1 = v15 | 1;
  if ( v16 != (_QWORD *)v28 )
    goto LABEL_25;
  return a1;
}
