// Function: sub_1584040
// Address: 0x1584040
//
__int64 __fastcall sub_1584040(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rbx
  char v11; // al
  __int64 v12; // r8
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  char v18; // al
  char v19; // dl
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  int v24; // ebx
  unsigned int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rdi
  _BYTE *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // [rsp+18h] [rbp-158h]
  __int64 v34; // [rsp+18h] [rbp-158h]
  __int64 v35; // [rsp+20h] [rbp-150h]
  __int64 v36; // [rsp+20h] [rbp-150h]
  __int64 v37; // [rsp+28h] [rbp-148h]
  __int64 v38; // [rsp+28h] [rbp-148h]
  __int64 v39; // [rsp+28h] [rbp-148h]
  _BYTE *v40; // [rsp+30h] [rbp-140h] BYREF
  __int64 v41; // [rsp+38h] [rbp-138h]
  _BYTE v42[304]; // [rsp+40h] [rbp-130h] BYREF

  if ( (unsigned __int8)sub_1593BB0(a1) )
    return a3;
  if ( (unsigned __int8)sub_1596070(a1) )
    return (__int64)a2;
  v6 = *(_BYTE *)(a1 + 16);
  if ( v6 == 8 )
  {
    v40 = v42;
    v41 = 0x1000000000LL;
    v7 = sub_16498A0(a1);
    v8 = sub_1644900(v7, 32);
    v9 = *(_QWORD *)(*a2 + 32);
    if ( (_DWORD)v9 )
    {
      v10 = 0;
      v35 = (unsigned int)v9;
      while ( 1 )
      {
        v15 = sub_15A0680(v8, v10, 0);
        v37 = sub_15A37D0(a2, v15, 0);
        v16 = sub_15A0680(v8, v10, 0);
        v12 = sub_15A37D0(a3, v16, 0);
        if ( v37 != v12 )
        {
          v17 = *(_QWORD *)(a1 + 24 * (v10 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          v18 = *(_BYTE *)(v17 + 16);
          if ( v18 == 9 )
          {
            if ( *(_BYTE *)(v37 + 16) == 9 )
              v12 = v37;
            v13 = (unsigned int)v41;
            if ( (unsigned int)v41 < HIDWORD(v41) )
              goto LABEL_12;
LABEL_18:
            v38 = v12;
            sub_16CD150(&v40, v42, 0, 8);
            v13 = (unsigned int)v41;
            v12 = v38;
            goto LABEL_12;
          }
          if ( v18 != 13 )
          {
            v14 = v41;
            v9 = *(_QWORD *)(*a2 + 32);
            goto LABEL_20;
          }
          v33 = v12;
          v11 = sub_1593BB0(v17);
          v12 = v33;
          if ( !v11 )
            v12 = v37;
        }
        v13 = (unsigned int)v41;
        if ( (unsigned int)v41 >= HIDWORD(v41) )
          goto LABEL_18;
LABEL_12:
        ++v10;
        *(_QWORD *)&v40[8 * v13] = v12;
        v14 = v41 + 1;
        LODWORD(v41) = v41 + 1;
        if ( v35 == v10 )
        {
          v9 = *(_QWORD *)(*a2 + 32);
          goto LABEL_20;
        }
      }
    }
    v14 = v41;
LABEL_20:
    if ( v14 == (_DWORD)v9 )
    {
      result = sub_15A01B0(v40, (unsigned int)v9);
      v30 = (unsigned __int64)v40;
      if ( v40 == v42 )
        return result;
      goto LABEL_59;
    }
    if ( v40 != v42 )
      _libc_free((unsigned __int64)v40);
    v6 = *(_BYTE *)(a1 + 16);
  }
  v19 = *((_BYTE *)a2 + 16);
  if ( v6 == 9 )
  {
    if ( v19 != 9 )
      return a3;
    return (__int64)a2;
  }
  if ( v19 == 9 )
    return a3;
  v20 = *(_BYTE *)(a3 + 16);
  if ( a2 == (__int64 *)a3 || v20 == 9 )
    return (__int64)a2;
  if ( v19 == 5 )
  {
    if ( *((_WORD *)a2 + 9) == 55 )
    {
      v22 = a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
      if ( v22 == a1 )
      {
        if ( v22 )
        {
          v23 = a3;
          a2 = (__int64 *)a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))];
          return sub_15A2DC0(a1, a2, v23, 0);
        }
      }
    }
    if ( v20 != 5 )
      return 0;
    goto LABEL_31;
  }
  if ( v20 != 5 )
  {
    if ( v20 != 7 || v19 != 7 )
      return 0;
    v24 = *(_DWORD *)(*a2 + 12);
    v36 = *a2;
    v40 = v42;
    v41 = 0x2000000000LL;
    if ( v24 )
    {
      v25 = 0;
      while ( 1 )
      {
        v27 = sub_15A0A60(a2, v25);
        v28 = sub_15A0A60(a3, v25);
        if ( v27 == v28 )
          goto LABEL_49;
        if ( *(_BYTE *)(v27 + 16) != 9 )
          break;
        v29 = (unsigned int)v41;
        if ( (unsigned int)v41 >= HIDWORD(v41) )
        {
          v34 = v28;
          sub_16CD150(&v40, v42, 0, 8);
          v29 = (unsigned int)v41;
          v28 = v34;
        }
        *(_QWORD *)&v40[8 * v29] = v28;
        LODWORD(v41) = v41 + 1;
LABEL_52:
        if ( ++v25 == v24 )
        {
          v31 = v40;
          v32 = (unsigned int)v41;
          goto LABEL_61;
        }
      }
      if ( *(_BYTE *)(v28 + 16) != 9 )
      {
        result = 0;
        goto LABEL_62;
      }
LABEL_49:
      v26 = (unsigned int)v41;
      if ( (unsigned int)v41 >= HIDWORD(v41) )
      {
        sub_16CD150(&v40, v42, 0, 8);
        v26 = (unsigned int)v41;
      }
      *(_QWORD *)&v40[8 * v26] = v27;
      LODWORD(v41) = v41 + 1;
      goto LABEL_52;
    }
    v31 = v42;
    v32 = 0;
LABEL_61:
    result = sub_159F090(v36, v31, v32);
LABEL_62:
    v30 = (unsigned __int64)v40;
    if ( v40 == v42 )
      return result;
LABEL_59:
    v39 = result;
    _libc_free(v30);
    return v39;
  }
LABEL_31:
  result = 0;
  if ( *(_WORD *)(a3 + 18) == 55 )
  {
    v21 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    if ( v21 == a1 && v21 )
    {
      v23 = *(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
      return sub_15A2DC0(a1, a2, v23, 0);
    }
    return 0;
  }
  return result;
}
