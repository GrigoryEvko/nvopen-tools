// Function: sub_14951F0
// Address: 0x14951f0
//
__int64 *__fastcall sub_14951F0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rsi
  __int64 *v12; // r15
  int v13; // r13d
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // r8
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // rdi
  __int64 *v21; // rax
  int v22; // r11d
  __int64 *v23; // r10
  int v24; // ecx
  int v25; // ecx
  __int64 *v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+10h] [rbp-90h] BYREF
  __int64 *v28; // [rsp+18h] [rbp-88h] BYREF
  __int64 v29; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v30; // [rsp+28h] [rbp-78h]
  __int64 *v31; // [rsp+30h] [rbp-70h]
  __int64 v32; // [rsp+38h] [rbp-68h]
  int v33; // [rsp+40h] [rbp-60h]
  _BYTE v34[88]; // [rsp+48h] [rbp-58h] BYREF

  v6 = sub_1494E70(a1, a2, a3, a4);
  v7 = *(_QWORD *)(a1 + 120);
  v8 = *(_QWORD *)(a1 + 112);
  v29 = 0;
  v30 = (__int64 *)v34;
  v31 = (__int64 *)v34;
  v32 = 4;
  v33 = 0;
  v26 = sub_1493080(v8, (__int64)v6, v7, (__int64)&v29, a3, a4);
  if ( v26 )
  {
    v9 = v31;
    if ( v31 == v30 )
      v10 = &v31[HIDWORD(v32)];
    else
      v10 = &v31[(unsigned int)v32];
    if ( v31 != v10 )
    {
      while ( 1 )
      {
        v11 = *v9;
        v12 = v9;
        if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v10 == ++v9 )
          goto LABEL_7;
      }
      if ( v9 != v10 )
      {
        do
        {
          sub_146E690(a1 + 128, v11);
          v21 = v12 + 1;
          if ( v12 + 1 == v10 )
            break;
          v11 = *v21;
          for ( ++v12; (unsigned __int64)*v21 >= 0xFFFFFFFFFFFFFFFELL; v12 = v21 )
          {
            if ( v10 == ++v21 )
              goto LABEL_7;
            v11 = *v21;
          }
        }
        while ( v10 != v12 );
      }
    }
LABEL_7:
    sub_14950C0(a1, a3, a4);
    v13 = *(_DWORD *)(a1 + 344);
    v14 = sub_146F1B0(*(_QWORD *)(a1 + 112), a2);
    v15 = *(_DWORD *)(a1 + 24);
    v27 = v14;
    if ( v15 )
    {
      v16 = *(_QWORD *)(a1 + 8);
      v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = (__int64 *)(v16 + 24LL * v17);
      v19 = *v18;
      if ( v14 == *v18 )
      {
LABEL_9:
        *((_DWORD *)v18 + 2) = v13;
        v18[2] = (__int64)v26;
        goto LABEL_10;
      }
      v22 = 1;
      v23 = 0;
      while ( v19 != -8 )
      {
        if ( v19 == -16 && !v23 )
          v23 = v18;
        v17 = (v15 - 1) & (v22 + v17);
        v18 = (__int64 *)(v16 + 24LL * v17);
        v19 = *v18;
        if ( v14 == *v18 )
          goto LABEL_9;
        ++v22;
      }
      v24 = *(_DWORD *)(a1 + 16);
      if ( v23 )
        v18 = v23;
      ++*(_QWORD *)a1;
      v25 = v24 + 1;
      if ( 4 * v25 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 20) - v25 > v15 >> 3 )
        {
LABEL_27:
          *(_DWORD *)(a1 + 16) = v25;
          if ( *v18 != -8 )
            --*(_DWORD *)(a1 + 20);
          *v18 = v14;
          *((_DWORD *)v18 + 2) = 0;
          v18[2] = 0;
          goto LABEL_9;
        }
LABEL_32:
        sub_146EB50(a1, v15);
        sub_145F250(a1, &v27, &v28);
        v18 = v28;
        v14 = v27;
        v25 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_27;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    v15 *= 2;
    goto LABEL_32;
  }
LABEL_10:
  if ( v31 != v30 )
    _libc_free((unsigned __int64)v31);
  return v26;
}
