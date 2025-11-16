// Function: sub_1999100
// Address: 0x1999100
//
__int64 __fastcall sub_1999100(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int8 a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // r8
  int v11; // eax
  bool v12; // al
  int v13; // eax
  __int16 v14; // ax
  __int64 v15; // r13
  int v16; // ebx
  __int64 v17; // r8
  __int64 result; // rax
  unsigned int v19; // r15d
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // r15
  __int64 *v26; // r8
  __int64 *v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // r15d
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r15d
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // r15d
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 *v42; // [rsp+0h] [rbp-B0h]
  __int64 v43; // [rsp+8h] [rbp-A8h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  char v47; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+18h] [rbp-98h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  _QWORD *v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v54; // [rsp+30h] [rbp-80h] BYREF
  __int64 v55; // [rsp+38h] [rbp-78h]
  _BYTE v56[112]; // [rsp+40h] [rbp-70h] BYREF

  if ( a1 == a2 )
  {
    v23 = sub_1456040(a1);
    return sub_145CF80((__int64)a3, v23, 1, 0);
  }
  if ( *(_WORD *)(a2 + 24) )
  {
    v14 = *(_WORD *)(a1 + 24);
    if ( !v14 )
      return 0;
LABEL_15:
    switch ( v14 )
    {
      case 7:
        if ( a4
          || (v29 = sub_1456040(**(_QWORD **)(a1 + 32)),
              v30 = sub_1456C90((__int64)a3, v29),
              v31 = (_QWORD *)sub_15E0530(a3[3]),
              v32 = sub_1644900(v31, v30 + 1),
              *(_WORD *)(sub_147B0D0((__int64)a3, a1, v32, 0) + 24) == 7) )
        {
          if ( *(_QWORD *)(a1 + 40) == 2 )
          {
            v19 = a4;
            v20 = sub_13A5BC0((_QWORD *)a1, (__int64)a3);
            v21 = sub_1999100(v20, a2, a3, a4);
            if ( v21 )
            {
              v22 = sub_1999100(**(_QWORD **)(a1 + 32), a2, a3, v19);
              if ( v22 )
                return sub_14799E0((__int64)a3, v22, v21, *(_QWORD *)(a1 + 48), 0);
            }
          }
        }
        return 0;
      case 4:
        if ( !a4 )
        {
          v34 = sub_1456040(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a1 + 40) - 1)));
          v35 = sub_1456C90((__int64)a3, v34);
          v36 = (_QWORD *)sub_15E0530(a3[3]);
          v37 = sub_1644900(v36, v35 + 1);
          if ( *(_WORD *)(sub_147B0D0((__int64)a3, a1, v37, 0) + 24) != 4 )
            return 0;
        }
        v24 = *(_QWORD **)(a1 + 32);
        v54 = (__int64 *)v56;
        v55 = 0x800000000LL;
        v51 = &v24[*(_QWORD *)(a1 + 40)];
        if ( v24 == v51 )
        {
LABEL_40:
          result = (__int64)sub_147DD40((__int64)a3, (__int64 *)&v54, 0, 0, a5, a6);
        }
        else
        {
          v25 = v24;
          while ( 1 )
          {
            result = sub_1999100(*v25, a2, a3, a4);
            v53 = result;
            if ( !result )
              break;
            ++v25;
            sub_1458920((__int64)&v54, &v53);
            if ( v51 == v25 )
              goto LABEL_40;
          }
        }
        break;
      case 5:
        if ( !a4 )
        {
          v38 = sub_1456040(**(_QWORD **)(a1 + 32));
          v39 = sub_1456C90((__int64)a3, v38) * *(_DWORD *)(a1 + 40);
          v40 = (_QWORD *)sub_15E0530(a3[3]);
          v41 = sub_1644900(v40, v39);
          if ( *(_WORD *)(sub_147B0D0((__int64)a3, a1, v41, 0) + 24) != 5 )
            return 0;
        }
        v26 = *(__int64 **)(a1 + 32);
        v54 = (__int64 *)v56;
        v55 = 0x400000000LL;
        v42 = &v26[*(_QWORD *)(a1 + 40)];
        result = 0;
        if ( v26 == v42 )
          return result;
        v47 = 0;
        v27 = v26;
        do
        {
          v28 = *v27;
          v53 = *v27;
          if ( !v47 )
          {
            v33 = sub_1999100(v28, a2, a3, a4);
            if ( v33 )
            {
              v53 = v33;
              v47 = 1;
            }
          }
          ++v27;
          sub_1458920((__int64)&v54, &v53);
        }
        while ( v42 != v27 );
        result = 0;
        if ( v47 )
          result = sub_147EE30(a3, &v54, 0, 0, a5, a6);
        break;
      default:
        return 0;
    }
    if ( v54 != (__int64 *)v56 )
    {
      v52 = result;
      _libc_free((unsigned __int64)v54);
      return v52;
    }
    return result;
  }
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_DWORD *)(v8 + 32);
  v10 = v8 + 24;
  if ( v9 <= 0x40 )
  {
    v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)(v8 + 24);
  }
  else
  {
    v43 = *(_QWORD *)(a2 + 32);
    v45 = v8 + 24;
    v11 = sub_16A58F0(v8 + 24);
    v10 = v45;
    v8 = v43;
    v12 = v9 == v11;
  }
  if ( v12 )
    return sub_13A5B60((__int64)a3, a1, a2, 0, 0);
  if ( v9 <= 0x40 )
  {
    if ( *(_QWORD *)(v8 + 24) != 1 )
      goto LABEL_9;
    return a1;
  }
  v44 = v8;
  v46 = v10;
  v13 = sub_16A57B0(v10);
  v10 = v46;
  if ( v9 - v13 <= 0x40 && **(_QWORD **)(v44 + 24) == 1 )
    return a1;
LABEL_9:
  v14 = *(_WORD *)(a1 + 24);
  if ( v14 )
    goto LABEL_15;
  v49 = v10;
  v15 = *(_QWORD *)(a1 + 32) + 24LL;
  sub_16AB4D0((__int64)&v54, v15, v10);
  v16 = v55;
  v17 = v49;
  if ( (unsigned int)v55 > 0x40 )
  {
    if ( v16 - (unsigned int)sub_16A57B0((__int64)&v54) > 0x40 || *v54 )
    {
      if ( v54 )
        j_j___libc_free_0_0(v54);
      return 0;
    }
    j_j___libc_free_0_0(v54);
    v17 = v49;
  }
  else if ( v54 )
  {
    return 0;
  }
  sub_16A9F90((__int64)&v54, v15, v17);
  result = sub_145CF40((__int64)a3, (__int64)&v54);
  if ( (unsigned int)v55 > 0x40 )
  {
    if ( v54 )
    {
      v50 = result;
      j_j___libc_free_0_0(v54);
      return v50;
    }
  }
  return result;
}
