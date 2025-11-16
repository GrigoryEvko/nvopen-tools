// Function: sub_E7F630
// Address: 0xe7f630
//
__int64 __fastcall sub_E7F630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r12d
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // r14
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64, unsigned int, __int64, unsigned int, unsigned int); // rax
  __int64 (__fastcall *v19)(__int64, __int64); // rax
  __int64 result; // rax
  char v21; // al
  int v22; // edx
  __m128i v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // edx
  int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // [rsp+8h] [rbp-128h]
  __m128i v31[2]; // [rsp+10h] [rbp-120h] BYREF
  char v32; // [rsp+30h] [rbp-100h]
  char v33; // [rsp+31h] [rbp-FFh]
  __m128i v34; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v35; // [rsp+60h] [rbp-D0h]
  __m128i v36[3]; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v37[2]; // [rsp+A0h] [rbp-90h] BYREF
  char v38; // [rsp+C0h] [rbp-70h]
  char v39; // [rsp+C1h] [rbp-6Fh]
  __m128i v40[2]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v41; // [rsp+F0h] [rbp-40h]

  v8 = a4;
  sub_E5CB20(*(_QWORD *)(a1 + 296), a2, a3, a4, a5, a6);
  if ( !(unsigned __int8)sub_EA1770(a2) )
    sub_EA1710(a2, 1);
  sub_EA15B0(a2, 1);
  if ( (unsigned int)sub_EA1780(a2) )
  {
    v21 = *(_BYTE *)(a2 + 9) & 0x70;
    if ( ((v21 - 48) & 0xE0) != 0 )
    {
      v27 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)(a2 + 24) = a3;
      v28 = (((unsigned __int8)v8 + 1) << 15) & 0xF8000;
      BYTE1(v28) |= 0x30u;
      *(_DWORD *)(a2 + 8) = v27 & 0xFFF00FFF | v28;
    }
    else if ( a3 != *(_QWORD *)(a2 + 24)
           || (v22 = (*(_DWORD *)(a2 + 8) >> 15) & 0x1F) == 0
           || (_BYTE)v22 - 1 != (_BYTE)v8
           || v21 == 64 )
    {
      v39 = 1;
      v37[0].m128i_i64[0] = (__int64)" redeclared as different type";
      v38 = 3;
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v29 = *(__int64 **)(a2 - 8);
        v23.m128i_i64[1] = *v29;
        v23.m128i_i64[0] = (__int64)(v29 + 3);
      }
      else
      {
        v23 = 0u;
      }
      v34 = v23;
      v31[0].m128i_i64[0] = (__int64)"Symbol: ";
      v35 = 261;
      v33 = 1;
      v32 = 3;
      sub_9C6370(v36, v31, &v34, v10, v11, v12);
      sub_9C6370(v40, v36, v37, v24, v25, v26);
      sub_C64D30((__int64)v40, 1u);
    }
  }
  else
  {
    v13 = **(_QWORD **)(a1 + 296);
    v37[0].m128i_i64[0] = (__int64)".bss";
    v39 = 1;
    v38 = 3;
    v41 = 257;
    v14 = sub_E71CB0(v13, (size_t *)v37, 8, 3u, 0, (__int64)v40, 0, -1, 0);
    v15 = *(unsigned int *)(a1 + 128);
    v30 = 0;
    v16 = v14;
    if ( (_DWORD)v15 )
    {
      v17 = *(_QWORD *)(a1 + 120) + 32 * v15 - 32;
      LODWORD(v15) = *(_DWORD *)(v17 + 8);
      v30 = *(_QWORD *)v17;
    }
    (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)a1 + 176LL))(a1, v16, 0);
    v18 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, unsigned int, unsigned int))(*(_QWORD *)a1 + 608LL);
    if ( v18 == sub_E7F5D0 )
    {
      if ( sub_E7E4B0(a1) )
        sub_C64ED0("Emitting values inside a locked bundle is forbidden", 1u);
      sub_E8B560(a1, v8, 0, 1, 0);
    }
    else
    {
      v18(a1, v8, 0, 1u, 0);
    }
    v19 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 208LL);
    if ( v19 == sub_E7F4D0 )
    {
      sub_E8DC70(a1, a2, 0);
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL) + 153LL) & 4) != 0 )
        sub_EA15B0(a2, 6);
    }
    else
    {
      ((void (__fastcall *)(__int64, __int64, _QWORD))v19)(a1, a2, 0);
    }
    sub_E99300(a1, a3);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 176LL))(a1, v30, (unsigned int)v15);
  }
  result = sub_E81A90(a3, *(_QWORD *)(a1 + 8), 0, 0);
  *(_QWORD *)(a2 + 32) = result;
  return result;
}
