// Function: sub_35D36B0
// Address: 0x35d36b0
//
__int64 __fastcall sub_35D36B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v5; // r13d
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  _DWORD *v16; // rax
  unsigned __int16 *v17; // rdx
  _DWORD *v18; // rsi
  unsigned __int16 *i; // rdi
  unsigned __int16 v20; // cx
  void (*v21)(void); // rax
  __int64 v22; // [rsp+0h] [rbp-90h]
  unsigned __int8 v23; // [rsp+Fh] [rbp-81h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  unsigned __int8 v25; // [rsp+18h] [rbp-78h]
  __int64 v26; // [rsp+20h] [rbp-70h]
  __m128i v28; // [rsp+30h] [rbp-60h] BYREF
  __int64 v29; // [rsp+40h] [rbp-50h]
  _DWORD *v30; // [rsp+48h] [rbp-48h]

  if ( (_BYTE)qword_50401A8
    && (v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16)),
        *(_QWORD *)(a1 + 200) = v3,
        (v23 = *(_BYTE *)(*(_QWORD *)(a2 + 48) + 40LL)) != 0)
    && (v22 = a2 + 320, v26 = *(_QWORD *)(a2 + 328), v26 != a2 + 320) )
  {
    v25 = 0;
    while ( 1 )
    {
      *(_QWORD *)(a1 + 208) = v3;
      *(_QWORD *)(a1 + 224) = 0;
      v5 = *(_DWORD *)(v3 + 16);
      if ( v5 < *(_DWORD *)(a1 + 264) >> 2 || v5 > *(_DWORD *)(a1 + 264) )
      {
        v6 = (__int64)_libc_calloc(v5, 1u);
        if ( !v6 && (v5 || (v6 = malloc(1u)) == 0) )
          sub_C64F00("Allocation failed", 1u);
        v7 = *(_QWORD *)(a1 + 256);
        *(_QWORD *)(a1 + 256) = v6;
        if ( v7 )
          _libc_free(v7);
        *(_DWORD *)(a1 + 264) = v5;
      }
      sub_3508720((_QWORD **)(a1 + 208), v26);
      v24 = *(_QWORD *)(v26 + 48);
      v8 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v24 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
LABEL_36:
        BUG();
      v9 = *(_QWORD *)v8;
      v10 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = v11;
          if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
            break;
          v9 = *(_QWORD *)v11;
        }
      }
      while ( v26 + 48 != v10 )
      {
        if ( *(_WORD *)(v10 + 68) == 28 )
        {
          v16 = sub_2E7B000(a2);
          v17 = *(unsigned __int16 **)(a1 + 216);
          v18 = v16;
          for ( i = &v17[*(_QWORD *)(a1 + 224)]; v17 != i; v16[v20 >> 5] |= 1 << v20 )
            v20 = *v17++;
          v21 = *(void (**)(void))(**(_QWORD **)(a1 + 200) + 248LL);
          if ( v21 != nullsub_1704 )
            v21();
          v30 = v18;
          v28.m128i_i64[0] = 13;
          v29 = 0;
          sub_2E8EAD0(v10, a2, &v28);
          v25 = v23;
        }
        sub_3508F10((_QWORD *)(a1 + 208), v10);
        v12 = (_QWORD *)(*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
        v13 = v12;
        if ( !v12 )
          goto LABEL_36;
        v10 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = *v12;
        if ( (v14 & 4) == 0 && (*((_BYTE *)v13 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
            v10 = v15;
            if ( (*(_BYTE *)(v15 + 44) & 4) == 0 )
              break;
            v14 = *(_QWORD *)v15;
          }
        }
      }
      v26 = *(_QWORD *)(v26 + 8);
      if ( v22 == v26 )
        break;
      v3 = *(_QWORD *)(a1 + 200);
    }
  }
  else
  {
    return 0;
  }
  return v25;
}
