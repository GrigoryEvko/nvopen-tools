// Function: sub_BA6240
// Address: 0xba6240
//
__int64 __fastcall sub_BA6240(__int64 a1, _BYTE *a2)
{
  __int64 *v2; // rax
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rsi
  unsigned int v8; // ecx
  __int64 v9; // r8
  __int64 v10; // r13
  unsigned int v11; // r8d
  _BYTE *v12; // rcx
  int v13; // r14d
  _QWORD *v14; // rdx
  __int64 v15; // rdi
  _BYTE *v16; // r10
  unsigned int v17; // esi
  _QWORD *v18; // rax
  _BYTE *v19; // r11
  __int64 *v20; // rsi
  _QWORD *v21; // rdx
  __int64 *v22; // rax
  int v23; // esi
  int v24; // ecx
  __int64 v25; // rdi
  __int64 v26; // r12
  unsigned int v27; // r9d
  __int64 v28; // rax
  int v29; // eax
  _BYTE *v30; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v31; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v30 = a2;
  result = *v2;
  v4 = *(_QWORD *)result;
  v5 = *(unsigned int *)(*(_QWORD *)result + 592LL);
  v6 = *(_QWORD *)(*(_QWORD *)result + 576LL);
  if ( (_DWORD)v5 )
  {
    v8 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    result = v6 + 16LL * v8;
    v9 = *(_QWORD *)result;
    if ( a1 == *(_QWORD *)result )
    {
LABEL_3:
      if ( result == v6 + 16 * v5 )
        return result;
      *(_BYTE *)(a1 + 7) &= ~8u;
      v10 = *(_QWORD *)(result + 8);
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(v4 + 584);
      ++*(_DWORD *)(v4 + 588);
      if ( *(_BYTE *)v10 != 2 )
      {
        if ( *v30 <= 0x15u )
          goto LABEL_6;
LABEL_27:
        sub_BA6110((const __m128i *)(v10 + 8), 0);
        if ( (*(_BYTE *)(v10 + 32) & 1) != 0 )
          return j_j___libc_free_0(v10, 144);
        goto LABEL_14;
      }
      if ( *v30 <= 0x15u )
      {
        v22 = sub_B98A20((__int64)v30, v6);
        sub_BA6110((const __m128i *)(v10 + 8), v22);
        if ( (*(_BYTE *)(v10 + 32) & 1) != 0 )
          return j_j___libc_free_0(v10, 144);
        goto LABEL_14;
      }
      if ( *(_BYTE *)a1 == 22 )
      {
        v25 = *(_QWORD *)(a1 + 24);
        if ( !v25 )
          goto LABEL_6;
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 40);
        if ( !v28 )
          goto LABEL_6;
        v25 = *(_QWORD *)(v28 + 72);
        if ( !v25 )
          goto LABEL_6;
      }
      if ( sub_B92180(v25) )
      {
        if ( sub_B921A0((__int64)v30) )
        {
          v26 = sub_B921A0(a1);
          if ( v26 != sub_B921A0((__int64)v30) )
            goto LABEL_27;
        }
      }
LABEL_6:
      v11 = *(_DWORD *)(v4 + 592);
      if ( v11 )
      {
        v12 = v30;
        v13 = 1;
        v14 = 0;
        v15 = *(_QWORD *)(v4 + 576);
        v16 = v30;
        v17 = (v11 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v18 = (_QWORD *)(v15 + 16LL * v17);
        v19 = (_BYTE *)*v18;
        if ( v30 == (_BYTE *)*v18 )
        {
LABEL_8:
          v20 = (__int64 *)v18[1];
          v21 = v18 + 1;
          if ( v20 )
          {
            sub_BA6110((const __m128i *)(v10 + 8), v20);
            if ( (*(_BYTE *)(v10 + 32) & 1) != 0 )
              return j_j___libc_free_0(v10, 144);
LABEL_14:
            sub_C7D6A0(*(_QWORD *)(v10 + 40), 24LL * *(unsigned int *)(v10 + 48), 8);
            return j_j___libc_free_0(v10, 144);
          }
LABEL_21:
          v12[7] |= 8u;
          result = (__int64)v30;
          *(_QWORD *)(v10 + 136) = v30;
          *v21 = v10;
          return result;
        }
        while ( v19 != (_BYTE *)-4096LL )
        {
          if ( v19 == (_BYTE *)-8192LL && !v14 )
            v14 = v18;
          v17 = (v11 - 1) & (v13 + v17);
          v18 = (_QWORD *)(v15 + 16LL * v17);
          v19 = (_BYTE *)*v18;
          if ( v30 == (_BYTE *)*v18 )
            goto LABEL_8;
          ++v13;
        }
        if ( !v14 )
          v14 = v18;
        v29 = *(_DWORD *)(v4 + 584);
        ++*(_QWORD *)(v4 + 568);
        v24 = v29 + 1;
        v31 = v14;
        if ( 4 * (v29 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(v4 + 588) - v24 > v11 >> 3 )
            goto LABEL_18;
          v23 = v11;
LABEL_17:
          sub_B98840(v4 + 568, v23);
          sub_B927C0(v4 + 568, (__int64 *)&v30, &v31);
          v16 = v30;
          v14 = v31;
          v24 = *(_DWORD *)(v4 + 584) + 1;
LABEL_18:
          *(_DWORD *)(v4 + 584) = v24;
          if ( *v14 != -4096 )
            --*(_DWORD *)(v4 + 588);
          *v14 = v16;
          v12 = v30;
          v21 = v14 + 1;
          *v21 = 0;
          goto LABEL_21;
        }
      }
      else
      {
        ++*(_QWORD *)(v4 + 568);
        v31 = 0;
      }
      v23 = 2 * v11;
      goto LABEL_17;
    }
    result = 1;
    while ( v9 != -4096 )
    {
      v27 = result + 1;
      v8 = (v5 - 1) & (result + v8);
      result = v6 + 16LL * v8;
      v9 = *(_QWORD *)result;
      if ( a1 == *(_QWORD *)result )
        goto LABEL_3;
      result = v27;
    }
  }
  return result;
}
