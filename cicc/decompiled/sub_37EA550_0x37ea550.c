// Function: sub_37EA550
// Address: 0x37ea550
//
__int64 __fastcall sub_37EA550(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r12
  unsigned __int64 v15; // rbx
  __int64 v16; // r8
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // rsi
  unsigned int v21; // eax
  __int64 v22; // rdi
  unsigned __int16 *v23; // rdx
  __int64 v24; // rdi
  void (*v25)(); // rax
  unsigned int v26; // [rsp-3Ch] [rbp-3Ch]

  result = *(_QWORD *)(a1 + 552);
  if ( *(_QWORD *)(a1 + 544) != result )
  {
    result = sub_B2D610(**(_QWORD **)(a1 + 200), 18);
    if ( !(_BYTE)result )
    {
      v7 = *(_QWORD *)(a1 + 216);
      *(_QWORD *)(a1 + 584) = 0;
      *(_QWORD *)(a1 + 568) = v7;
      v8 = *(_DWORD *)(v7 + 16);
      v9 = *(_DWORD *)(a1 + 624) >> 2;
      if ( v8 < (unsigned int)v9 || v8 > *(_DWORD *)(a1 + 624) )
      {
        v10 = (__int64)_libc_calloc(v8, 1u);
        if ( !v10 && (v8 || (v10 = malloc(1u)) == 0) )
          sub_C64F00("Allocation failed", 1u);
        v11 = *(_QWORD *)(a1 + 616);
        *(_QWORD *)(a1 + 616) = v10;
        if ( v11 )
          _libc_free(v11);
        *(_DWORD *)(a1 + 624) = v8;
      }
      v12 = a2 + 48;
      sub_35085F0((_QWORD *)(a1 + 568), a2, v9, v4, v5, v6);
      v13 = *(_QWORD *)(a1 + 552);
      v14 = *(_QWORD *)(v13 - 16);
      v26 = *(_DWORD *)(v13 - 8);
      v15 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v15 )
        BUG();
      result = *(_QWORD *)v15;
      if ( (*(_QWORD *)v15 & 4) == 0 && (*(_BYTE *)(v15 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          result &= 0xFFFFFFFFFFFFFFF8LL;
          v15 = result;
          if ( (*(_BYTE *)(result + 44) & 4) == 0 )
            break;
          result = *(_QWORD *)result;
        }
      }
      while ( v12 != v15 )
      {
        sub_3508F10((_QWORD *)(a1 + 568), v15);
        if ( v15 == v14 )
        {
          v19 = *(_QWORD *)(a1 + 584);
          v20 = *(unsigned __int16 *)(*(_QWORD *)(v15 + 32) + 40LL * v26 + 8);
          v21 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 616) + v20);
          if ( v21 >= (unsigned int)v19 )
            goto LABEL_26;
          v22 = *(_QWORD *)(a1 + 576);
          while ( 1 )
          {
            v23 = (unsigned __int16 *)(v22 + 2LL * v21);
            if ( *v23 == (_DWORD)v20 )
              break;
            v21 += 256;
            if ( (unsigned int)v19 <= v21 )
              goto LABEL_26;
          }
          if ( v23 == (unsigned __int16 *)(v22 + 2 * v19) )
          {
LABEL_26:
            v24 = *(_QWORD *)(a1 + 208);
            v25 = *(void (**)())(*(_QWORD *)v24 + 1248LL);
            if ( v25 != nullsub_1686 )
              ((void (__fastcall *)(__int64, __int64, _QWORD, _QWORD, __int64, _QWORD))v25)(
                v24,
                v14,
                v26,
                *(_QWORD *)(a1 + 216),
                v16,
                (unsigned int)v19);
          }
          *(_QWORD *)(a1 + 552) -= 16LL;
          result = *(_QWORD *)(a1 + 552);
          if ( result == *(_QWORD *)(a1 + 544) )
            return result;
          v14 = *(_QWORD *)(result - 16);
          v26 = *(_DWORD *)(result - 8);
        }
        v17 = (__int64 *)(*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v17;
        if ( !v17 )
          BUG();
        v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
        result = *v17;
        if ( (result & 4) == 0 && (*((_BYTE *)v18 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            result &= 0xFFFFFFFFFFFFFFF8LL;
            v15 = result;
            if ( (*(_BYTE *)(result + 44) & 4) == 0 )
              break;
            result = *(_QWORD *)result;
          }
        }
      }
    }
  }
  return result;
}
