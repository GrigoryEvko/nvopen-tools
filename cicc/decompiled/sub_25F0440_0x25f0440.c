// Function: sub_25F0440
// Address: 0x25f0440
//
__int64 __fastcall sub_25F0440(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned int v9; // r13d
  unsigned int v11; // eax
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  unsigned __int64 v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // rbx
  __int64 v19; // r14
  unsigned __int64 v20; // rbx
  unsigned int v23; // r15d
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  char v28; // al
  __int64 v29; // rsi
  __int64 v30; // rdi
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rax
  char v34; // al
  unsigned __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rsi
  __int64 *v38; // rdi
  unsigned __int64 v39; // [rsp+8h] [rbp-58h]
  unsigned int v40; // [rsp+8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 v42[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( a5 )
  {
    v6 = *a1;
    v7 = sub_FDD2C0(a5, a2, 0);
    v42[1] = v8;
    v42[0] = v7;
    if ( (_BYTE)v8 )
    {
      LOBYTE(v11) = sub_D84450(v6, v42[0]);
      v9 = v11;
      if ( (_BYTE)v11 )
        return v9;
    }
    goto LABEL_3;
  }
  v20 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v20 == a2 + 48 )
    goto LABEL_75;
  if ( !v20 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
LABEL_75:
    BUG();
  if ( *(_BYTE *)(v20 - 24) != 31 || !(unsigned __int8)sub_BC8C50(v20 - 24, &v41, v42) || !(v42[0] + v41) )
  {
    v28 = *(_BYTE *)(a4 + 28);
    goto LABEL_42;
  }
  v39 = v42[0] + v41;
  v23 = sub_F02DD0(v41, v42[0] + v41);
  v25 = (unsigned int)sub_F02DD0(v42[0], v39);
  if ( a3 < v23 )
    goto LABEL_33;
  v28 = *(_BYTE *)(a4 + 28);
  v37 = *(_QWORD *)(v20 - 56);
  if ( !v28 )
    goto LABEL_67;
  v24 = *(__int64 **)(a4 + 8);
  v26 = *(unsigned int *)(a4 + 20);
  v38 = &v24[v26];
  if ( v24 == v38 )
  {
LABEL_65:
    if ( (unsigned int)v26 < *(_DWORD *)(a4 + 16) )
    {
      v26 = (unsigned int)(v26 + 1);
      *(_DWORD *)(a4 + 20) = v26;
      *v38 = v37;
      ++*(_QWORD *)a4;
LABEL_33:
      v28 = *(_BYTE *)(a4 + 28);
      goto LABEL_34;
    }
LABEL_67:
    v40 = v25;
    sub_C8CC70(a4, v37, (__int64)v24, v25, v26, v27);
    v28 = *(_BYTE *)(a4 + 28);
    v25 = v40;
    goto LABEL_34;
  }
  while ( v37 != *v24 )
  {
    if ( v38 == ++v24 )
      goto LABEL_65;
  }
LABEL_34:
  if ( a3 < (unsigned int)v25 )
    goto LABEL_42;
  v29 = *(_QWORD *)(v20 - 88);
  if ( !v28 )
  {
LABEL_68:
    sub_C8CC70(a4, v29, (__int64)v24, v25, v26, v27);
    v28 = *(_BYTE *)(a4 + 28);
    goto LABEL_42;
  }
  v24 = *(__int64 **)(a4 + 8);
  v30 = *(unsigned int *)(a4 + 20);
  v25 = (__int64)&v24[v30];
  if ( v24 == (__int64 *)v25 )
  {
LABEL_39:
    if ( (unsigned int)v30 < *(_DWORD *)(a4 + 16) )
    {
      *(_DWORD *)(a4 + 20) = v30 + 1;
      *(_QWORD *)v25 = v29;
      v28 = *(_BYTE *)(a4 + 28);
      ++*(_QWORD *)a4;
      goto LABEL_42;
    }
    goto LABEL_68;
  }
  while ( v29 != *v24 )
  {
    if ( (__int64 *)v25 == ++v24 )
      goto LABEL_39;
  }
LABEL_42:
  if ( v28 )
  {
    v31 = *(_QWORD **)(a4 + 8);
    v32 = &v31[*(unsigned int *)(a4 + 20)];
    if ( v31 != v32 )
    {
      if ( a2 == *v31 )
        return 1;
      while ( v32 != ++v31 )
      {
        if ( a2 == *v31 )
          return 1;
      }
    }
  }
  else if ( sub_C8CA60(a4, a2) )
  {
    return 1;
  }
LABEL_3:
  v9 = (unsigned __int8)qword_4FF1D88;
  if ( !(_BYTE)qword_4FF1D88 )
    return 0;
  v12 = sub_AA4FF0(a2);
  if ( !v12 )
    BUG();
  v13 = (unsigned int)*(unsigned __int8 *)(v12 - 24) - 39;
  if ( (unsigned int)v13 > 0x38 || (v14 = 0x100060000000001LL, !_bittest64(&v14, v13)) )
  {
    v15 = a2 + 48;
    v16 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v17 = (_QWORD *)v16;
    if ( a2 + 48 == v16 )
      goto LABEL_74;
    if ( !v16 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30 > 0xA )
LABEL_74:
      BUG();
    if ( *(_BYTE *)(v16 - 24) != 35 )
    {
      v18 = *(_QWORD *)(a2 + 56);
      if ( v18 == v15 )
      {
LABEL_52:
        if ( (unsigned int)*((unsigned __int8 *)v17 - 24) - 30 > 0xA )
          BUG();
        if ( (unsigned int)sub_B46E30((__int64)(v17 - 3)) )
          return 0;
        v34 = *((_BYTE *)v17 - 24);
        if ( v34 == 30 || v34 == 33 )
          return 0;
        if ( *(_QWORD **)(v17[2] + 56LL) != v17 )
        {
          v35 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v35 )
          {
            if ( *(_BYTE *)(v35 - 24) == 85 )
            {
              v36 = v35 - 24;
              if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(v35 + 48), 36) )
                return (unsigned int)sub_B49560(v36, 36) ^ 1;
              return 0;
            }
          }
        }
      }
      else
      {
        v19 = 0x8000000000041LL;
        while ( 1 )
        {
          if ( !v18 )
            BUG();
          if ( (unsigned __int8)(*(_BYTE *)(v18 - 24) - 34) <= 0x33u
            && _bittest64(&v19, (unsigned int)*(unsigned __int8 *)(v18 - 24) - 34)
            && ((unsigned __int8)sub_A73ED0((_QWORD *)(v18 + 48), 5) || (unsigned __int8)sub_B49560(v18 - 24, 5))
            && ((*(_BYTE *)(v18 - 17) & 0x20) == 0 || !sub_B91C10(v18 - 24, 31)) )
          {
            break;
          }
          v18 = *(_QWORD *)(v18 + 8);
          if ( v18 == v15 )
          {
            v33 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            v17 = (_QWORD *)v33;
            if ( v18 == v33 )
              BUG();
            if ( !v33 )
              BUG();
            goto LABEL_52;
          }
        }
      }
    }
    return 1;
  }
  return v9;
}
