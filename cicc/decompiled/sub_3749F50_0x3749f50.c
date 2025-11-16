// Function: sub_3749F50
// Address: 0x3749f50
//
__int64 __fastcall sub_3749F50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  unsigned int v7; // eax
  __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v11; // r15
  unsigned int v12; // ebx
  unsigned int v13; // r10d
  _BYTE **v14; // rdx
  _BYTE *v15; // rbx
  unsigned int v16; // eax
  unsigned int v17; // r15d
  unsigned int v18; // r10d
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int v21; // eax
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // r8
  unsigned int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r9
  __int64 (*v29)(); // rax
  unsigned int v30; // edx
  unsigned int v31; // ecx
  _QWORD *v32; // r8
  unsigned int v33; // eax
  bool v34; // al
  unsigned __int64 v35; // r8
  unsigned int v36; // [rsp+0h] [rbp-60h]
  __int64 (__fastcall *v37)(__int64, __int64, unsigned int, __int64); // [rsp+0h] [rbp-60h]
  unsigned int v38; // [rsp+0h] [rbp-60h]
  _BYTE **v39; // [rsp+8h] [rbp-58h]
  unsigned int v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+8h] [rbp-58h]
  unsigned __int64 v42; // [rsp+8h] [rbp-58h]
  char v43[8]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int16 v44; // [rsp+18h] [rbp-48h]

  v5 = a3;
  v7 = sub_30097B0(*(_QWORD *)(a2 + 8), 1, a3, a4, a5);
  if ( (unsigned __int16)v7 <= 1u )
    return 0;
  v11 = a1[16];
  HIWORD(v12) = HIWORD(v7);
  v13 = v7;
  if ( !*(_QWORD *)(v11 + 8LL * (unsigned __int16)v7 + 112) )
  {
    v41 = v8;
    if ( (_WORD)v7 != 2 || v5 - 186 > 2 )
      return 0;
    LOWORD(v12) = 2;
    v37 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
    v26 = sub_BD5C60(a2);
    if ( v37 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)v43, v11, v26, v12, v41);
      v13 = v44;
    }
    else
    {
      v13 = v37(v11, v26, v12, v41);
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v14 = *(_BYTE ***)(a2 - 8);
  else
    v14 = (_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v15 = *v14;
  v39 = v14;
  if ( **v14 == 17 && *(_BYTE *)a2 > 0x1Cu )
  {
    v36 = v13;
    LOBYTE(v16) = sub_B46D50((unsigned __int8 *)a2);
    v13 = v36;
    v9 = v16;
    if ( (_BYTE)v16 )
    {
      v31 = sub_3746830(a1, (__int64)v39[4]);
      if ( v31 )
      {
        v32 = (_QWORD *)*((_QWORD *)v15 + 3);
        if ( *((_DWORD *)v15 + 8) > 0x40u )
          v32 = (_QWORD *)*v32;
        v33 = sub_3749CE0((__int64 **)a1, v36, v5, v31, (unsigned __int64)v32, v36);
        if ( v33 )
        {
          sub_3742B00((__int64)a1, (_BYTE *)a2, v33, 1);
          return v9;
        }
      }
      return 0;
    }
  }
  v40 = v13;
  v17 = sub_3746830(a1, (__int64)v15);
  if ( !v17 )
    return 0;
  v18 = v40;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v19 = *(_QWORD *)(a2 - 8);
  else
    v19 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v20 = *(_QWORD *)(v19 + 32);
  if ( *(_BYTE *)v20 != 17 )
  {
    v27 = sub_3746830(a1, v20);
    v28 = v27;
    if ( v27 )
    {
      v29 = *(__int64 (**)())(*a1 + 72);
      if ( v29 != sub_3740EF0 )
      {
        v30 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, _QWORD, _QWORD, __int64))v29)(
                a1,
                v40,
                v40,
                v5,
                v17,
                v28);
        if ( v30 )
        {
          v9 = 1;
          sub_3742B00((__int64)a1, (_BYTE *)a2, v30, 1);
          return v9;
        }
      }
    }
    return 0;
  }
  v21 = *(_DWORD *)(v20 + 32);
  v22 = *(unsigned __int64 **)(v20 + 24);
  if ( v21 <= 0x40 )
  {
    v23 = 0;
    if ( v21 )
      v23 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
  }
  else
  {
    v23 = *v22;
  }
  if ( v5 == 59 )
  {
    if ( (unsigned __int8)(*(_BYTE *)a2 - 42) <= 0x11u )
    {
      v38 = v40;
      v42 = v23;
      v34 = sub_B44E60(a2);
      v23 = v42;
      v18 = v38;
      if ( v42 )
      {
        if ( v34 && (v42 & (v42 - 1)) == 0 )
        {
          _BitScanReverse64(&v35, v42);
          v5 = 191;
          v23 = (int)(63 - (v35 ^ 0x3F));
        }
      }
    }
  }
  else if ( v5 == 62 && (unsigned __int8)(*(_BYTE *)a2 - 42) <= 0x11u && v23 && (v23 & (v23 - 1)) == 0 )
  {
    --v23;
    v5 = 186;
  }
  v24 = v17;
  v9 = 0;
  v25 = sub_3749CE0((__int64 **)a1, v18, v5, v24, v23, v18);
  if ( v25 )
  {
    sub_3742B00((__int64)a1, (_BYTE *)a2, v25, 1);
    return 1;
  }
  return v9;
}
