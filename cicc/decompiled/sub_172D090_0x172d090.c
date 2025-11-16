// Function: sub_172D090
// Address: 0x172d090
//
__int64 __fastcall sub_172D090(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 result; // rax
  char v5; // dl
  __int64 v6; // r14
  char v7; // si
  __int64 v8; // rdx
  int v9; // eax
  int v10; // eax
  __int64 **v11; // rdx
  __int64 v12; // r15
  unsigned int v13; // r13d
  __int64 v14; // rax
  char v15; // al
  __int64 v16; // rax
  int v17; // edx
  int v18; // edx
  __int64 **v19; // rax
  __int64 *v20; // r13
  __int64 v21; // r15
  unsigned int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r15
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  int v33; // edx
  int v34; // edx
  __int64 **v35; // rax
  __int64 *v36; // [rsp-90h] [rbp-90h]
  __int64 v37; // [rsp-90h] [rbp-90h]
  __int64 v38[2]; // [rsp-88h] [rbp-88h] BYREF
  __int16 v39; // [rsp-78h] [rbp-78h]
  __int64 *v40; // [rsp-68h] [rbp-68h]
  __int64 v41; // [rsp-60h] [rbp-60h]
  __int64 v42[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)(a2 + 16) != 51 )
    return 0;
  if ( !sub_1642F90(*(_QWORD *)a2, 16) )
    return 0;
  v3 = *(_QWORD *)(a2 - 48);
  result = 0;
  v5 = *(_BYTE *)(v3 + 16);
  if ( (unsigned __int8)(v5 - 35) <= 0x11u )
  {
    v6 = *(_QWORD *)(a2 - 24);
    v7 = *(_BYTE *)(v6 + 16);
    if ( (unsigned __int8)(v7 - 35) <= 0x11u )
    {
      if ( v5 != 50 )
      {
        if ( v7 != 50 )
          return 0;
        v6 = *(_QWORD *)(a2 - 48);
        v3 = *(_QWORD *)(a2 - 24);
      }
      v8 = *(_QWORD *)(v3 - 48);
      v9 = *(unsigned __int8 *)(v8 + 16);
      if ( (unsigned __int8)v9 > 0x17u )
      {
        v10 = v9 - 24;
      }
      else
      {
        if ( (_BYTE)v9 != 5 )
          return 0;
        v10 = *(unsigned __int16 *)(v8 + 18);
      }
      if ( v10 != 36 )
        return 0;
      v11 = (*(_BYTE *)(v8 + 23) & 0x40) != 0
          ? *(__int64 ***)(v8 - 8)
          : (__int64 **)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
      v36 = *v11;
      if ( !*v11 )
        return 0;
      v12 = *(_QWORD *)(v3 - 24);
      if ( *(_BYTE *)(v12 + 16) != 13 )
        return 0;
      v13 = *(_DWORD *)(v12 + 32);
      if ( v13 > 0x40 )
      {
        if ( v13 - (unsigned int)sub_16A57B0(v12 + 24) > 0x40 )
          return 0;
        v14 = **(_QWORD **)(v12 + 24);
      }
      else
      {
        v14 = *(_QWORD *)(v12 + 24);
      }
      if ( v14 != 255 || !sub_1642F90(*v36, 32) )
        return 0;
      v15 = *(_BYTE *)(v6 + 16);
      if ( v15 == 47 )
      {
        v32 = *(_QWORD *)(v6 - 48);
        v33 = *(unsigned __int8 *)(v32 + 16);
        if ( (unsigned __int8)v33 > 0x17u )
        {
          v34 = v33 - 24;
        }
        else
        {
          if ( (_BYTE)v33 != 5 )
            return 0;
          v34 = *(unsigned __int16 *)(v32 + 18);
        }
        if ( v34 != 36 )
          return 0;
        v35 = (*(_BYTE *)(v32 + 23) & 0x40) != 0
            ? *(__int64 ***)(v32 - 8)
            : (__int64 **)(v32 - 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF));
        v20 = *v35;
        if ( !*v35 )
          return 0;
        v21 = *(_QWORD *)(v6 - 24);
        if ( *(_BYTE *)(v21 + 16) != 13 )
          return 0;
      }
      else
      {
        if ( v15 != 5 || *(_WORD *)(v6 + 18) != 23 )
          return 0;
        v16 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v17 = *(unsigned __int8 *)(v16 + 16);
        if ( (unsigned __int8)v17 > 0x17u )
        {
          v18 = v17 - 24;
        }
        else
        {
          if ( (_BYTE)v17 != 5 )
            return 0;
          v18 = *(unsigned __int16 *)(v16 + 18);
        }
        if ( v18 != 36 )
          return 0;
        v19 = (*(_BYTE *)(v16 + 23) & 0x40) != 0
            ? *(__int64 ***)(v16 - 8)
            : (__int64 **)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
        v20 = *v19;
        if ( !*v19 )
          return 0;
        v21 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v21 + 16) != 13 )
          return 0;
      }
      v22 = *(_DWORD *)(v21 + 32);
      if ( v22 > 0x40 )
      {
        if ( v22 - (unsigned int)sub_16A57B0(v21 + 24) > 0x40 )
          return 0;
        v23 = **(_QWORD **)(v21 + 24);
      }
      else
      {
        v23 = *(_QWORD *)(v21 + 24);
      }
      if ( v23 != 8 || !sub_1642F90(*v20, 32) )
        return 0;
      v24 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) + 40LL), 4227, 0, 0);
      v42[1] = (__int64)v20;
      v25 = v24;
      v40 = v42;
      v42[0] = (__int64)v36;
      v41 = 0x300000002LL;
      v26 = (_QWORD *)sub_16498A0(a2);
      v27 = sub_1643350(v26);
      v42[2] = sub_159C470(v27, 64, 0);
      v28 = *(_QWORD *)(a1 + 8);
      v39 = 259;
      v38[0] = (__int64)"prmtCall";
      v29 = *(_QWORD *)(v25 + 24);
      LODWORD(v41) = 3;
      v30 = sub_172C570(v28, v29, v25, v42, 3, v38, 0);
      v31 = *(_QWORD *)a2;
      v39 = 257;
      result = sub_15FDBD0(36, v30, v31, (__int64)v38, 0);
      if ( v40 != v42 )
      {
        v37 = result;
        _libc_free((unsigned __int64)v40);
        return v37;
      }
    }
  }
  return result;
}
