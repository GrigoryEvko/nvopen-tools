// Function: sub_385F290
// Address: 0x385f290
//
char __fastcall sub_385F290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 v6; // r15
  unsigned __int8 v9; // al
  unsigned __int8 v10; // dl
  char result; // al
  __int64 v12; // rdx
  int v13; // edx
  _QWORD *v14; // r14
  unsigned int v15; // esi
  __int64 v16; // r9
  char v17; // r8
  int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // ebx
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rsi
  unsigned __int64 v25; // rsi
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r15
  __int64 v35; // r13
  __int64 *v36; // rbx
  __int64 v37; // rax
  int v38; // eax
  unsigned __int64 v39; // rax
  _QWORD *v40; // rax
  int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // [rsp+0h] [rbp-90h]
  char v44; // [rsp+8h] [rbp-88h]
  char v45; // [rsp+8h] [rbp-88h]
  char v46; // [rsp+8h] [rbp-88h]
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+8h] [rbp-88h]
  unsigned __int64 v51; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v52; // [rsp+18h] [rbp-78h]
  unsigned __int64 v53; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v54; // [rsp+28h] [rbp-68h]
  unsigned __int64 v55; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v56; // [rsp+38h] [rbp-58h]
  __int64 v57[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v58[8]; // [rsp+50h] [rbp-40h] BYREF

  v6 = 0;
  v9 = *(_BYTE *)(a1 + 16);
  if ( v9 > 0x17u )
  {
    if ( (unsigned __int8)(v9 - 54) <= 1u )
    {
      v6 = *(_QWORD *)(a1 - 24);
    }
    else if ( v9 == 78 )
    {
      v22 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v22 + 16) )
      {
        v23 = *(_DWORD *)(v22 + 36);
        if ( v23 == 4085 || v23 == 4057 )
        {
          v6 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        }
        else if ( v23 == 4503 || v23 == 4492 )
        {
          v6 = *(_QWORD *)(a1 + 24 * (2LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        }
      }
    }
  }
  v10 = *(_BYTE *)(a2 + 16);
  if ( v10 <= 0x17u )
    return 0;
  if ( v10 == 54 || v10 == 55 )
  {
    v14 = *(_QWORD **)(a2 - 24);
    if ( !v14 )
      return 0;
  }
  else
  {
    result = 0;
    if ( v10 != 78 )
      return result;
    v12 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v12 + 16) )
      return result;
    v13 = *(_DWORD *)(v12 + 36);
    if ( v13 == 4057 || v13 == 4085 )
    {
      v14 = *(_QWORD **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( !v14 )
        return result;
    }
    else
    {
      result = v13 == 4503 || v13 == 4492;
      if ( !result )
        return result;
      v14 = *(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( !v14 )
        return 0;
    }
  }
  v15 = sub_385B460(a1);
  result = (v14 == (_QWORD *)v6) | (v15 != (unsigned int)sub_385B460(v16) || v6 == 0);
  if ( result )
    return 0;
  if ( !v17 || *(_QWORD *)v6 == *v14 )
  {
    v18 = sub_15A95A0(a3, v15);
    v19 = 1;
    v20 = 8 * v18;
    v21 = *(_QWORD *)(*(_QWORD *)v6 + 24LL);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v21 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v37 = *(_QWORD *)(v21 + 32);
          v21 = *(_QWORD *)(v21 + 24);
          v19 *= v37;
          continue;
        case 1:
          v24 = 16;
          break;
        case 2:
          v24 = 32;
          break;
        case 3:
        case 9:
          v24 = 64;
          break;
        case 4:
          v24 = 80;
          break;
        case 5:
        case 6:
          v24 = 128;
          break;
        case 7:
          v50 = v19;
          v41 = sub_15A9520(a3, 0);
          v19 = v50;
          v24 = (unsigned int)(8 * v41);
          break;
        case 0xB:
          v24 = *(_DWORD *)(v21 + 8) >> 8;
          break;
        case 0xD:
          v49 = v19;
          v40 = (_QWORD *)sub_15A9930(a3, v21);
          v19 = v49;
          v24 = 8LL * *v40;
          break;
        case 0xE:
          v43 = v19;
          v48 = *(_QWORD *)(v21 + 32);
          v39 = sub_12BE0A0(a3, *(_QWORD *)(v21 + 24));
          v19 = v43;
          v24 = 8 * v48 * v39;
          break;
        case 0xF:
          v47 = v19;
          v38 = sub_15A9520(a3, *(_DWORD *)(v21 + 8) >> 8);
          v19 = v47;
          v24 = (unsigned int)(8 * v38);
          break;
      }
      break;
    }
    v52 = v20;
    v25 = (unsigned __int64)(v24 * v19 + 7) >> 3;
    if ( v20 > 0x40 )
    {
      sub_16A4EF0((__int64)&v51, v25, 0);
      v54 = v20;
      sub_16A4EF0((__int64)&v53, 0, 0);
      v56 = v20;
      sub_16A4EF0((__int64)&v55, 0, 0);
    }
    else
    {
      v54 = v20;
      v53 = 0;
      v55 = 0;
      v56 = v20;
      v51 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & v25;
    }
    v26 = sub_164A410(v6, a3, (__int64)&v53);
    v27 = sub_164A410((__int64)v14, a3, (__int64)&v55);
    v28 = sub_145CF40(a4, (__int64)&v53);
    v29 = sub_145CF40(a4, (__int64)&v55);
    v30 = sub_14806B0(a4, v29, v28, 0, 0);
    v31 = v30;
    if ( *(_WORD *)(v30 + 24) )
      BUG();
    if ( v26 == v27 )
    {
      v42 = *(_QWORD *)(v30 + 32);
      if ( *(_DWORD *)(v42 + 32) <= 0x40u )
        result = *(_QWORD *)(v42 + 24) == v51;
      else
        result = sub_16A5220(v42 + 24, (const void **)&v51);
    }
    else
    {
      v32 = sub_145CF40(a4, (__int64)&v51);
      v33 = sub_14806B0(a4, v32, v31, 0, 0);
      v34 = sub_146F1B0(a4, v26);
      v35 = sub_146F1B0(a4, v27);
      v58[1] = v33;
      v57[0] = (__int64)v58;
      v58[0] = v34;
      v57[1] = 0x200000002LL;
      v36 = sub_147DD40(a4, v57, 0, 0, a5, a6);
      if ( (_QWORD *)v57[0] != v58 )
        _libc_free(v57[0]);
      result = v35 == (_QWORD)v36;
    }
    if ( v56 > 0x40 && v55 )
    {
      v44 = result;
      j_j___libc_free_0_0(v55);
      result = v44;
    }
    if ( v54 > 0x40 && v53 )
    {
      v45 = result;
      j_j___libc_free_0_0(v53);
      result = v45;
    }
    if ( v52 > 0x40 )
    {
      if ( v51 )
      {
        v46 = result;
        j_j___libc_free_0_0(v51);
        return v46;
      }
    }
  }
  return result;
}
