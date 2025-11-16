// Function: sub_26278B0
// Address: 0x26278b0
//
_BYTE *__fastcall sub_26278B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *result; // rax
  const void *v5; // rdx
  size_t v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rcx
  int v18; // ebx
  unsigned int v19; // ecx
  __int64 v20; // r9
  __int64 v21; // [rsp+0h] [rbp-E0h]
  _BYTE *v22; // [rsp+8h] [rbp-D8h]
  __int64 v23; // [rsp+10h] [rbp-D0h]
  __int64 v24; // [rsp+18h] [rbp-C8h]
  _BYTE *v25; // [rsp+20h] [rbp-C0h]
  __int64 v26; // [rsp+28h] [rbp-B8h]
  __int64 v27; // [rsp+30h] [rbp-B0h]
  const void *v29[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v30; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v31; // [rsp+58h] [rbp-88h]
  _QWORD v32[2]; // [rsp+60h] [rbp-80h] BYREF
  __int128 v33; // [rsp+70h] [rbp-70h] BYREF
  __int128 v34; // [rsp+80h] [rbp-60h]
  __int128 v35; // [rsp+90h] [rbp-50h]
  __int64 v36; // [rsp+A0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v2 != 24 )
    sub_C64ED0("Second argument of llvm.type.test must be metadata", 1u);
  result = *(_BYTE **)(v2 + 24);
  v27 = (__int64)result;
  if ( !*result )
  {
    v29[0] = (const void *)sub_B91420((__int64)result);
    v29[1] = v5;
    v6 = (size_t)v5;
    v7 = a1[2];
    v8 = sub_B2F650((__int64)v29[0], (__int64)v5);
    v9 = *(_QWORD *)(v7 + 224);
    v10 = (_QWORD *)(v7 + 216);
    v11 = v8;
    if ( !v9 )
      goto LABEL_9;
    while ( 1 )
    {
      while ( v11 > *(_QWORD *)(v9 + 32) )
      {
        v9 = *(_QWORD *)(v9 + 24);
        if ( !v9 )
          goto LABEL_9;
      }
      v12 = *(_QWORD **)(v9 + 16);
      if ( v11 >= *(_QWORD *)(v9 + 32) )
        break;
      v10 = (_QWORD *)v9;
      v9 = *(_QWORD *)(v9 + 16);
      if ( !v12 )
        goto LABEL_9;
    }
    v14 = *(_QWORD **)(v9 + 24);
    if ( v14 )
    {
      do
      {
        while ( 1 )
        {
          v15 = v14[2];
          v16 = v14[3];
          if ( v11 < v14[4] )
            break;
          v14 = (_QWORD *)v14[3];
          if ( !v16 )
            goto LABEL_20;
        }
        v10 = v14;
        v14 = (_QWORD *)v14[2];
      }
      while ( v15 );
    }
LABEL_20:
    while ( v12 )
    {
      while ( 1 )
      {
        v17 = v12[3];
        if ( v11 <= v12[4] )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v17 )
          goto LABEL_23;
      }
      v9 = (__int64)v12;
      v12 = (_QWORD *)v12[2];
    }
LABEL_23:
    if ( (_QWORD *)v9 == v10 )
    {
LABEL_9:
      v36 = 0;
      v33 = 0;
      v34 = 0;
      v35 = 0;
      goto LABEL_10;
    }
    while ( v6 != *(_QWORD *)(v9 + 48) || v6 && memcmp(*(const void **)(v9 + 40), v29[0], v6) )
    {
      v9 = sub_220EF30(v9);
      if ( (_QWORD *)v9 == v10 )
        goto LABEL_9;
    }
    v18 = *(_DWORD *)(v9 + 56);
    v31 = (__int64 *)v29;
    v30 = a1;
    v32[0] = a1;
    v32[1] = &v30;
    if ( v18 )
    {
      v25 = sub_2623AF0(a1, (__int64 *)v29, "global_addr", (void *)0xB);
      if ( ((unsigned int)(v18 - 1) <= 1 || v18 == 4)
        && (v24 = sub_2623BF0(v32, "align", (void *)5, *(_QWORD *)(v9 + 64), 8, a1[8]),
            v23 = sub_2623BF0(v32, "size_m1", (void *)7, *(_QWORD *)(v9 + 72), *(_DWORD *)(v9 + 60), a1[14]),
            v18 == 1) )
      {
        v22 = sub_2623AF0(v30, v31, "byte_array", (void *)0xA);
        v21 = sub_2623BF0(v32, "bit_mask", (void *)8, *(unsigned __int8 *)(v9 + 80), 8, a1[9]);
      }
      else if ( v18 == 2 )
      {
        v19 = *(_DWORD *)(v9 + 60);
        if ( v19 <= 5 )
          v20 = a1[11];
        else
          v20 = a1[13];
        v26 = sub_2623BF0(v32, "inline_bits", (void *)0xB, *(_QWORD *)(v9 + 88), 1 << v19, v20);
      }
    }
    LODWORD(v33) = v18;
    *((_QWORD *)&v33 + 1) = v25;
    *(_QWORD *)&v34 = v24;
    *((_QWORD *)&v34 + 1) = v23;
    *(_QWORD *)&v35 = v22;
    *((_QWORD *)&v35 + 1) = v21;
    result = (_BYTE *)v26;
    v36 = v26;
    if ( v18 != 5 )
    {
      if ( v18 )
      {
        result = (_BYTE *)sub_26267C0((__int64)a1, v27, a2, (__int64)&v33);
        v13 = (__int64)result;
LABEL_11:
        if ( v13 )
        {
          sub_BD84D0(a2, v13);
          return (_BYTE *)sub_B43D60((_QWORD *)a2);
        }
        return result;
      }
LABEL_10:
      result = (_BYTE *)sub_ACD720(*(__int64 **)*a1);
      v13 = (__int64)result;
      goto LABEL_11;
    }
  }
  return result;
}
