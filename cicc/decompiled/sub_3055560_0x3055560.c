// Function: sub_3055560
// Address: 0x3055560
//
char __fastcall sub_3055560(__int128 a1, _QWORD *a2, char **a3, __int64 a4, __int64 a5)
{
  int v7; // eax
  char result; // al
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v18; // edx
  __int64 *v19; // rax
  int v20; // edx
  __int64 v21; // rcx
  int v22; // edx
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  char *v27; // r15
  unsigned __int64 v28; // rax
  char *v29; // r14
  _QWORD *v30; // rcx
  __int64 v31; // rdx
  _QWORD *v32; // rax
  char *v33; // r14
  __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int128 v37; // [rsp+0h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+10h] [rbp-90h] BYREF
  __int64 v39; // [rsp+18h] [rbp-88h]
  unsigned __int16 v40; // [rsp+20h] [rbp-80h] BYREF
  __int64 v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+30h] [rbp-70h] BYREF
  char v43; // [rsp+38h] [rbp-68h]
  __int64 v44; // [rsp+40h] [rbp-60h]
  __int64 v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+50h] [rbp-50h] BYREF
  __int64 v47; // [rsp+58h] [rbp-48h]
  char v48; // [rsp+60h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 24);
  v37 = a1;
  if ( v7 == 548 || v7 == 551 )
  {
    if ( *a2 )
    {
      if ( *a2 != (_QWORD)a1 )
        return 0;
    }
    else
    {
      *a2 = a1;
    }
    v9 = *(unsigned __int16 *)(a1 + 96);
    v10 = *(_QWORD *)(a1 + 104);
    v38 = v9;
    v39 = v10;
    if ( (_WORD)v9 )
    {
      v11 = word_4456580[v9 - 1];
      v12 = 0;
    }
    else
    {
      v11 = sub_3009970((__int64)&v38, *((__int64 *)&a1 + 1), v10, (__int64)a3, a4);
    }
    v40 = v11;
    v41 = v12;
    if ( v11 )
    {
      if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
        goto LABEL_49;
      v14 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
      LOBYTE(v13) = byte_444C4A0[16 * v11 - 8];
    }
    else
    {
      v44 = sub_3007260((__int64)&v40);
      v14 = v44;
      v45 = v13;
    }
    LOBYTE(v47) = v13;
    v46 = v14 * DWORD2(v37);
    if ( (char *)sub_CA1930(&v46) != *a3 )
      return 0;
    if ( !v40 )
    {
      v15 = sub_3007260((__int64)&v40);
      v46 = v15;
      v47 = v16;
LABEL_15:
      v42 = v15;
      v43 = v16;
      *a3 += sub_CA1930(&v42);
      return 1;
    }
    if ( v40 != 1 && (unsigned __int16)(v40 - 504) > 7u )
    {
      v16 = 16LL * (v40 - 1);
      v15 = *(_QWORD *)&byte_444C4A0[v16];
      LOBYTE(v16) = byte_444C4A0[v16 + 8];
      goto LABEL_15;
    }
LABEL_49:
    BUG();
  }
  sub_3055350((__int64)&v46, a4, (__m128i *)&v37, (__int64)a3, a4, a5);
  if ( !v48 )
    return 0;
  v18 = *(_DWORD *)(v37 + 24);
  if ( v18 <= 215 )
  {
    if ( v18 > 212 )
      return sub_3055560(**(_QWORD **)(v37 + 40), *(_QWORD *)(*(_QWORD *)(v37 + 40) + 8LL), a2, a3, a4);
    if ( v18 != 186 )
      return 0;
    v19 = *(__int64 **)(v37 + 40);
    v20 = *(_DWORD *)(*v19 + 24);
    if ( v20 == 11 || v20 == 35 )
    {
      v23 = v19[5];
      v24 = *((unsigned int *)v19 + 12);
      v36 = *(_QWORD *)(*v19 + 96);
      v26 = *(_QWORD *)(v36 + 24);
      if ( *(_DWORD *)(v36 + 32) <= 0x40u )
        goto LABEL_26;
    }
    else
    {
      v21 = v19[5];
      v22 = *(_DWORD *)(v21 + 24);
      if ( v22 != 11 && v22 != 35 )
        return 0;
      v23 = *v19;
      v24 = *((unsigned int *)v19 + 2);
      v25 = *(_QWORD *)(v21 + 96);
      v26 = *(_QWORD *)(v25 + 24);
      if ( *(_DWORD *)(v25 + 32) <= 0x40u )
      {
LABEL_26:
        v27 = *a3;
        if ( (unsigned __int8)sub_3055560(v23, v24, a2, a3, a4) )
        {
          v28 = v26 + 1;
          if ( v26 != -1 && (v26 & v28) == 0 )
          {
            _BitScanReverse64(&v28, v28);
            return *a3 - v27 == 63 - ((unsigned int)v28 ^ 0x3F);
          }
        }
        return 0;
      }
    }
    v26 = *(_QWORD *)v26;
    goto LABEL_26;
  }
  result = 0;
  if ( v18 == 536 )
  {
    v29 = *a3;
    if ( !(unsigned __int8)sub_3055560(
                             *(_QWORD *)(*(_QWORD *)(v37 + 40) + 40LL),
                             *(_QWORD *)(*(_QWORD *)(v37 + 40) + 48LL),
                             a2,
                             a3,
                             a4) )
      return 0;
    v30 = *(_QWORD **)(v37 + 40);
    v31 = *(_QWORD *)(v30[10] + 96LL);
    v32 = *(_QWORD **)(v31 + 24);
    if ( *(_DWORD *)(v31 + 32) > 0x40u )
      v32 = (_QWORD *)*v32;
    v33 = &v29[(_QWORD)v32];
    if ( *a3 != v33 || !(unsigned __int8)sub_3055560(*v30, v30[1], a2, a3, a4) )
      return 0;
    v34 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v37 + 40) + 120LL) + 96LL);
    v35 = *(_QWORD **)(v34 + 24);
    if ( *(_DWORD *)(v34 + 32) > 0x40u )
      v35 = (_QWORD *)*v35;
    return *a3 == &v33[(_QWORD)v35];
  }
  return result;
}
