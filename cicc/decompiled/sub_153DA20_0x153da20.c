// Function: sub_153DA20
// Address: 0x153da20
//
__int64 __fastcall sub_153DA20(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 v9; // rsi
  __int64 v10; // r12
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // edx
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r15
  __int64 *v20; // r14
  __int64 v21; // rdx
  int v22; // ecx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  int v26; // edx
  int v27; // eax
  int v28; // ecx
  __int128 v29; // xmm0
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // r8
  unsigned __int64 v35; // r14
  int v36; // edx
  int v37; // eax
  bool v38; // zf
  __int128 v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  unsigned __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+40h] [rbp-60h]
  __int128 v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+60h] [rbp-40h]

  result = (__int64)a2 - a1;
  v42 = (unsigned __int64)a2;
  v41 = a3;
  if ( (__int64)a2 - a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_20;
  }
  v40 = a8;
  v39 = a7;
  while ( 2 )
  {
    v9 = *(_QWORD *)(a1 + 16);
    --v41;
    v10 = a1 + 16 * (result >> 5);
    v44 = v39;
    v45 = v40;
    v11 = sub_153D4C0((__int64 *)&v44, v9, *(_QWORD *)v10);
    v12 = *(_QWORD *)(v42 - 16);
    if ( !v11 )
    {
      if ( !sub_153D4C0((__int64 *)&v44, *(_QWORD *)(a1 + 16), v12) )
      {
        v35 = v42;
        v38 = sub_153D4C0((__int64 *)&v44, *(_QWORD *)v10, *(_QWORD *)(v42 - 16)) == 0;
        v13 = *(_QWORD *)a1;
        if ( v38 )
          goto LABEL_7;
LABEL_26:
        *(_QWORD *)a1 = *(_QWORD *)(v35 - 16);
        v36 = *(_DWORD *)(v35 - 8);
        *(_QWORD *)(v35 - 16) = v13;
        v37 = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v36;
        *(_DWORD *)(v35 - 8) = v37;
        v16 = *(_QWORD *)(a1 + 16);
        v17 = *(_QWORD *)a1;
        goto LABEL_8;
      }
LABEL_18:
      v16 = *(_QWORD *)a1;
      v17 = *(_QWORD *)(a1 + 16);
      v27 = *(_DWORD *)(a1 + 8);
      v28 = *(_DWORD *)(a1 + 24);
      *(_QWORD *)a1 = v17;
      *(_QWORD *)(a1 + 16) = v16;
      *(_DWORD *)(a1 + 8) = v28;
      *(_DWORD *)(a1 + 24) = v27;
      goto LABEL_8;
    }
    if ( !sub_153D4C0((__int64 *)&v44, *(_QWORD *)v10, v12) )
    {
      v35 = v42;
      if ( sub_153D4C0((__int64 *)&v44, *(_QWORD *)(a1 + 16), *(_QWORD *)(v42 - 16)) )
      {
        v13 = *(_QWORD *)a1;
        goto LABEL_26;
      }
      goto LABEL_18;
    }
    v13 = *(_QWORD *)a1;
LABEL_7:
    *(_QWORD *)a1 = *(_QWORD *)v10;
    v14 = *(_DWORD *)(v10 + 8);
    *(_QWORD *)v10 = v13;
    v15 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v14;
    *(_DWORD *)(v10 + 8) = v15;
    v16 = *(_QWORD *)(a1 + 16);
    v17 = *(_QWORD *)a1;
LABEL_8:
    v18 = a1 + 16;
    v19 = v42;
    v44 = v39;
    v45 = v40;
    while ( 1 )
    {
      v20 = (__int64 *)v18;
      if ( sub_153D4C0((__int64 *)&v44, v16, v17) )
        goto LABEL_9;
      do
      {
        v21 = *(_QWORD *)(v19 - 16);
        v19 -= 16LL;
      }
      while ( sub_153D4C0((__int64 *)&v44, *(_QWORD *)a1, v21) );
      if ( v18 >= v19 )
        break;
      v25 = *(_QWORD *)v18;
      *(_QWORD *)v18 = *(_QWORD *)v19;
      v26 = *(_DWORD *)(v19 + 8);
      *(_QWORD *)v19 = v25;
      LODWORD(v25) = *(_DWORD *)(v18 + 8);
      *(_DWORD *)(v18 + 8) = v26;
      *(_DWORD *)(v19 + 8) = v25;
LABEL_9:
      v17 = *(_QWORD *)a1;
      v16 = *(_QWORD *)(v18 + 16);
      v18 += 16LL;
    }
    sub_153DA20(v18, v42, v41, v22, v23, v24, a7, a8);
    result = v18 - a1;
    if ( (__int64)(v18 - a1) > 256 )
    {
      if ( v41 )
      {
        v42 = v18;
        continue;
      }
LABEL_20:
      v29 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
      v30 = result >> 4;
      v31 = ((result >> 4) - 2) >> 1;
      v45 = a8;
      v44 = v29;
      v43 = a8;
      sub_153D840(a1, v31, result >> 4, *(_QWORD *)(a1 + 16 * v31), *(_QWORD *)(a1 + 16 * v31 + 8), a6, v29, a8);
      do
      {
        --v31;
        sub_153D840(a1, v31, v30, *(_QWORD *)(a1 + 16 * v31), *(_QWORD *)(a1 + 16 * v31 + 8), v32, v44, v45);
      }
      while ( v31 );
      do
      {
        v20 -= 2;
        v33 = *v20;
        v34 = v20[1];
        *v20 = *(_QWORD *)a1;
        *((_DWORD *)v20 + 2) = *(_DWORD *)(a1 + 8);
        result = sub_153D840(a1, 0, ((__int64)v20 - a1) >> 4, v33, v34, v32, v29, v43);
      }
      while ( (__int64)v20 - a1 > 16 );
    }
    return result;
  }
}
