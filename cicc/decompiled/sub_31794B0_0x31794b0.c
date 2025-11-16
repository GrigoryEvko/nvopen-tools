// Function: sub_31794B0
// Address: 0x31794b0
//
unsigned __int64 __fastcall sub_31794B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rax
  unsigned __int64 result; // rax
  __int64 v5; // r15
  _BYTE *v6; // rdi
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r10
  unsigned int v11; // esi
  _BYTE *v12; // rcx
  __int64 v13; // r11
  _BYTE *v14; // rcx
  _BYTE *v15; // rcx
  _BYTE *v16; // rcx
  int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  unsigned int v25; // edx
  __int64 *v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rcx
  _BYTE *v33; // rcx
  int v34; // esi
  int v35; // r8d
  _BYTE *v36; // rcx
  _BYTE *v37; // rcx
  __int64 v38; // [rsp+18h] [rbp-78h]
  unsigned __int64 v39; // [rsp+18h] [rbp-78h]
  _BYTE *v40; // [rsp+20h] [rbp-70h] BYREF
  __int64 v41; // [rsp+28h] [rbp-68h]
  _BYTE v42[96]; // [rsp+30h] [rbp-60h] BYREF

  v2 = *(_QWORD **)(a2 - 8);
  v3 = *(_QWORD **)(a1 + 240);
  if ( *v3 != *v2 )
    return 0;
  v5 = a1;
  v6 = (_BYTE *)v3[1];
  if ( *v6 != 17 )
    return 0;
  v8 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
  v9 = v8 >> 2;
  if ( !(v8 >> 2) )
  {
    v32 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_38:
    if ( v32 != 2 )
    {
      if ( v32 != 3 )
      {
        if ( v32 != 1 )
          goto LABEL_45;
        goto LABEL_41;
      }
      v36 = (_BYTE *)v2[4 * (unsigned int)(2 * (v9 + 1))];
      if ( v36 && v6 == v36 )
        goto LABEL_16;
      ++v9;
    }
    v37 = (_BYTE *)v2[4 * (unsigned int)(2 * (v9 + 1))];
    if ( v37 && v6 == v37 )
      goto LABEL_16;
    ++v9;
LABEL_41:
    v33 = (_BYTE *)v2[4 * (unsigned int)(2 * v9 + 2)];
    if ( v33 )
    {
      if ( v6 == v33 && v8 != v9 )
      {
        v17 = v9;
        if ( v9 != 4294967294LL )
          goto LABEL_17;
      }
    }
    goto LABEL_45;
  }
  v10 = 4 * v9;
  v11 = 2;
  v9 = 0;
  while ( 1 )
  {
    v13 = v9 + 1;
    v16 = (_BYTE *)v2[4 * v11];
    if ( v16 )
    {
      if ( v6 == v16 )
        break;
    }
    v12 = (_BYTE *)v2[4 * v11 + 8];
    if ( v12 && v6 == v12 )
    {
LABEL_36:
      v9 = v13;
      break;
    }
    v13 = v9 + 3;
    v14 = (_BYTE *)v2[4 * v11 + 16];
    if ( v14 && v6 == v14 )
    {
      v9 += 2;
      break;
    }
    v9 += 4;
    v15 = (_BYTE *)v2[4 * (unsigned int)(2 * v9)];
    if ( v15 && v6 == v15 )
      goto LABEL_36;
    v11 += 8;
    if ( v9 == v10 )
    {
      v32 = v8 - v9;
      goto LABEL_38;
    }
  }
LABEL_16:
  v17 = v9;
  if ( v8 != v9 )
  {
LABEL_17:
    v18 = 4LL * (unsigned int)(2 * v17 + 3);
    goto LABEL_18;
  }
LABEL_45:
  v18 = 4;
LABEL_18:
  v19 = 0;
  v20 = v5;
  v38 = v2[v18];
  v40 = v42;
  v41 = 0x600000000LL;
  if ( !v8 )
    goto LABEL_34;
  while ( 1 )
  {
    v21 = 4;
    if ( (_DWORD)v19 != -2 )
      v21 = 4LL * (unsigned int)(2 * v19 + 3);
    v22 = v2[v21];
    if ( v38 != v22 && (unsigned __int8)sub_2A64220(*(__int64 **)(v20 + 56), v2[v21]) )
    {
      v23 = *(unsigned int *)(v20 + 120);
      v24 = *(_QWORD *)(v20 + 104);
      if ( (_DWORD)v23 )
      {
        v25 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v26 = (__int64 *)(v24 + 8LL * v25);
        v27 = *v26;
        if ( *v26 == v22 )
        {
LABEL_28:
          if ( v26 != (__int64 *)(v24 + 8 * v23) )
            goto LABEL_20;
        }
        else
        {
          v34 = 1;
          while ( v27 != -4096 )
          {
            v35 = v34 + 1;
            v25 = (v23 - 1) & (v34 + v25);
            v26 = (__int64 *)(v24 + 8LL * v25);
            v27 = *v26;
            if ( v22 == *v26 )
              goto LABEL_28;
            v34 = v35;
          }
        }
      }
      if ( (unsigned __int8)sub_3175050(v20, *(_QWORD *)(a2 + 40), v22) )
      {
        v30 = (unsigned int)v41;
        v31 = (unsigned int)v41 + 1LL;
        if ( v31 > HIDWORD(v41) )
        {
          sub_C8D5F0((__int64)&v40, v42, v31, 8u, v28, v29);
          v30 = (unsigned int)v41;
        }
        *(_QWORD *)&v40[8 * v30] = v22;
        LODWORD(v41) = v41 + 1;
      }
    }
LABEL_20:
    if ( ++v19 == v8 )
      break;
    v2 = *(_QWORD **)(a2 - 8);
  }
  v5 = v20;
LABEL_34:
  result = sub_3178E50(v5, (__int64)&v40);
  if ( v40 != v42 )
  {
    v39 = result;
    _libc_free((unsigned __int64)v40);
    return v39;
  }
  return result;
}
