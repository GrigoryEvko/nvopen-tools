// Function: sub_1F6A5A0
// Address: 0x1f6a5a0
//
void __fastcall sub_1F6A5A0(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int16 v5; // ax
  __int64 v6; // rax
  unsigned int v7; // ebx
  char (__fastcall *v8)(__int64, __int64); // rax
  __int16 v9; // ax
  __int64 *v10; // rbx
  __int64 *v11; // r12
  __int64 v12; // rdi
  __int16 v13; // ax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 *v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // rax
  int v21; // r8d
  int v22; // r9d
  const __m128i *v23; // r15
  const __m128i *v24; // r12
  const __m128i *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // [rsp+10h] [rbp-B0h]
  _QWORD *v31; // [rsp+20h] [rbp-A0h]
  __m128i v33; // [rsp+30h] [rbp-90h] BYREF
  __int64 v34; // [rsp+40h] [rbp-80h]
  __int64 v35; // [rsp+48h] [rbp-78h]
  __int64 *v36; // [rsp+60h] [rbp-60h] BYREF
  __int64 v37; // [rsp+68h] [rbp-58h]
  _BYTE v38[80]; // [rsp+70h] [rbp-50h] BYREF

  v36 = (__int64 *)v38;
  v37 = 0x400000000LL;
  v29 = *(_QWORD **)(a1 + 328);
  if ( v29 == (_QWORD *)(a1 + 320) )
    return;
  do
  {
    v31 = v29 + 3;
    v3 = sub_1DD5EE0((__int64)v29);
    if ( v29 + 3 != (_QWORD *)v3 )
    {
      v4 = v3;
      while ( 1 )
      {
        v5 = *(_WORD *)(v4 + 46);
        if ( (v5 & 4) != 0 || (v5 & 8) == 0 )
          v6 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL) >> 3) & 1LL;
        else
          LOBYTE(v6) = sub_1E15D00(v4, 8u, 1);
        v7 = 0;
        if ( (_BYTE)v6 )
        {
          v7 = 28;
          if ( !HIBYTE(a3) && **(unsigned __int16 **)(v4 + 16) != *(_DWORD *)(a2 + 48) )
            v7 = 0;
        }
        v8 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 1008LL);
        if ( v8 != sub_1F3AA70 )
          break;
        v9 = *(_WORD *)(v4 + 46);
        if ( (v9 & 4) != 0 || (v9 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL) & 8LL) == 0 )
            goto LABEL_16;
        }
        else if ( !sub_1E15D00(v4, 8u, 1) )
        {
          goto LABEL_16;
        }
        v13 = *(_WORD *)(v4 + 46);
        if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
          v14 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL) >> 4) & 1LL;
        else
          LOBYTE(v14) = sub_1E15D00(v4, 0x10u, 1);
        if ( (_BYTE)v14 )
          goto LABEL_35;
LABEL_16:
        if ( v7 )
        {
          v15 = (unsigned __int64)v7 << 6;
          goto LABEL_37;
        }
LABEL_17:
        if ( (*(_BYTE *)v4 & 4) != 0 )
        {
          v4 = *(_QWORD *)(v4 + 8);
          if ( v31 == (_QWORD *)v4 )
            goto LABEL_19;
        }
        else
        {
          while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
            v4 = *(_QWORD *)(v4 + 8);
          v4 = *(_QWORD *)(v4 + 8);
          if ( v31 == (_QWORD *)v4 )
            goto LABEL_19;
        }
      }
      if ( !v8(a2, v4) )
        goto LABEL_16;
LABEL_35:
      if ( (_BYTE)a3 )
      {
        v15 = 1920;
LABEL_37:
        v16 = (__int64 *)(v4 + 64);
        v17 = v29[7];
        v18 = *(_QWORD *)(a2 + 8) + v15;
        if ( (*(_BYTE *)(v4 + 46) & 4) != 0 )
        {
          v19 = (__int64)sub_1E0B640(v17, v18, v16, 0);
          sub_1DD6E10((__int64)v29, (__int64 *)v4, v19);
        }
        else
        {
          v19 = (__int64)sub_1E0B640(v17, v18, v16, 0);
          sub_1DD5BA0(v29 + 2, v19);
          v27 = *(_QWORD *)v4;
          v28 = *(_QWORD *)v19;
          *(_QWORD *)(v19 + 8) = v4;
          v27 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v19 = v27 | v28 & 7;
          *(_QWORD *)(v27 + 8) = v19;
          *(_QWORD *)v4 = v19 | *(_QWORD *)v4 & 7LL;
        }
        v20 = **(unsigned __int16 **)(v4 + 16);
        v34 = 0;
        v33.m128i_i64[0] = 1;
        v35 = v20;
        sub_1E1A9C0(v19, v17, &v33);
        v23 = *(const __m128i **)(v4 + 32);
        v24 = (const __m128i *)((char *)v23 + 40 * *(unsigned int *)(v4 + 40));
        while ( v24 != v23 )
        {
          v25 = v23;
          v23 = (const __m128i *)((char *)v23 + 40);
          sub_1E1A9C0(v19, v17, v25);
        }
        v26 = (unsigned int)v37;
        if ( (unsigned int)v37 >= HIDWORD(v37) )
        {
          sub_16CD150((__int64)&v36, v38, 0, 8, v21, v22);
          v26 = (unsigned int)v37;
        }
        v36[v26] = v4;
        LODWORD(v37) = v37 + 1;
        goto LABEL_17;
      }
      goto LABEL_16;
    }
LABEL_19:
    v29 = (_QWORD *)v29[1];
  }
  while ( (_QWORD *)(a1 + 320) != v29 );
  v10 = v36;
  v11 = &v36[(unsigned int)v37];
  if ( v36 != v11 )
  {
    do
    {
      v12 = *v10++;
      sub_1E16240(v12);
    }
    while ( v11 != v10 );
    v11 = v36;
  }
  if ( v11 != (__int64 *)v38 )
    _libc_free((unsigned __int64)v11);
}
