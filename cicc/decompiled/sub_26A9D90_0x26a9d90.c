// Function: sub_26A9D90
// Address: 0x26a9d90
//
__int64 __fastcall sub_26A9D90(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  unsigned __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  char v14; // al
  __int64 v15; // rax
  _BYTE **v16; // r12
  __int64 v17; // r13
  __int64 result; // rax
  _BYTE **v19; // r15
  unsigned int v20; // edx
  __int64 v21; // rax
  char v22; // al
  unsigned __int8 *v23; // rax
  __int64 v24; // [rsp-8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  unsigned __int64 v29; // [rsp+28h] [rbp-38h]

  v2 = (__int64 *)(a1 + 72);
  v27 = 0;
  v5 = sub_250D070((_QWORD *)(a1 + 72));
  v26 = v5 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v6 = sub_25952D0(a2, v5 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 1, 0, 1);
  v25 = v6;
  if ( v6
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v6 + 112LL))(
         v6,
         "ompx_spmd_amenable",
         18)
    || !(unsigned __int8)sub_B46490(v5)
    || *(_BYTE *)v5 == 85
    && (v21 = *(_QWORD *)(v5 - 32)) != 0
    && !*(_BYTE *)v21
    && *(_QWORD *)(v21 + 24) == *(_QWORD *)(v5 + 80)
    && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
  {
    v22 = *(_BYTE *)(a1 + 401);
    *(_BYTE *)(a1 + 96) = 1;
    *(_BYTE *)(a1 + 400) = v22;
    *(_BYTE *)(a1 + 336) = *(_BYTE *)(a1 + 337);
    *(_BYTE *)(a1 + 240) = *(_BYTE *)(a1 + 241);
    *(_BYTE *)(a1 + 112) = *(_BYTE *)(a1 + 113);
    result = *(unsigned __int8 *)(a1 + 177);
    *(_BYTE *)(a1 + 176) = result;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 80);
    v8 = *(_QWORD *)(a1 + 72);
    v29 = v5;
    v26 = a2;
    v27 = &v25;
    v28 = a1;
    v9 = sub_251C7D0(a2, v8, v7, a1, 1, 0, 1);
    v10 = v24;
    if ( v9
      && ((v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL), v11 != sub_2534F50)
        ? (v12 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64))v11)(v9, v8))
        : (v12 = (_BYTE *)(v9 + 88)),
          (v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL), v13 != sub_2505E30)
        ? (v14 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64))v13)(v12, v8, v10))
        : (v14 = v12[9]),
          v14 && !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v9 + 120LL))(v9, v8, v10)) )
    {
      v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 112LL))(v9);
      v16 = *(_BYTE ***)(v15 + 32);
      v17 = v15;
      result = *(unsigned int *)(v15 + 40);
      v19 = &v16[result];
      v20 = result;
      if ( v19 != v16 )
      {
        while ( 1 )
        {
          result = sub_26A9950((__int64)&v26, *v16, v20);
          if ( *(_BYTE *)(a1 + 96) )
            break;
          if ( v19 == ++v16 )
            break;
          v20 = *(_DWORD *)(v17 + 40);
        }
      }
    }
    else
    {
      v23 = sub_250CBE0(v2, v8);
      return sub_26A9950((__int64)&v26, v23, 1u);
    }
  }
  return result;
}
