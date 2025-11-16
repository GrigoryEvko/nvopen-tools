// Function: sub_3740A90
// Address: 0x3740a90
//
unsigned __int64 __fastcall sub_3740A90(__int64 *a1, __int64 a2)
{
  __int16 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // r14
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // rdi
  const void *v16; // rax
  size_t v17; // rdx
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  _QWORD *v22; // r15
  __int64 v23; // rax
  _QWORD *i; // r13
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v28; // rax
  int v29; // edi
  __int64 v30; // rcx
  int v31; // edi
  unsigned int v32; // edx
  unsigned __int8 **v33; // rax
  unsigned __int8 *v34; // r8
  __int64 v35; // rdx
  int v36; // eax
  int v37; // r9d
  __int64 v38; // [rsp+8h] [rbp-38h]

  v4 = sub_AF18C0(a2);
  v5 = sub_A777F0(0x30u, a1 + 11);
  v6 = v5;
  if ( v5 )
  {
    *(_WORD *)(v5 + 28) = v4;
    *(_QWORD *)v5 = v5 | 4;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = -1;
    *(_BYTE *)(v5 + 30) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
  }
  v7 = a2 - 16;
  sub_324C3F0((__int64)a1, (unsigned __int8 *)a2, v5);
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(unsigned __int8 **)(*(_QWORD *)(a2 - 32) + 8LL);
    v10 = *v9;
    if ( (_BYTE)v10 == 21 )
    {
LABEL_5:
      v11 = sub_324DC40(a1, (__int64)v9);
      goto LABEL_6;
    }
  }
  else
  {
    v9 = *(unsigned __int8 **)(v7 - 8LL * ((v8 >> 2) & 0xF) + 8);
    v10 = *v9;
    if ( (_BYTE)v10 == 21 )
      goto LABEL_5;
  }
  if ( (_BYTE)v10 == 22 )
  {
    v11 = sub_324DE20(a1, (__int64)v9);
    goto LABEL_6;
  }
  if ( (_BYTE)v10 == 18 )
  {
    if ( !sub_3734FE0((__int64)a1) || (unsigned __int8)sub_321F6A0(a1[26], v9) )
      v28 = a1[27] + 400;
    else
      v28 = (__int64)(a1 + 84);
    v29 = *(_DWORD *)(v28 + 24);
    v30 = *(_QWORD *)(v28 + 8);
    if ( v29 )
    {
      v31 = v29 - 1;
      v32 = v31 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v33 = (unsigned __int8 **)(v30 + 16LL * v32);
      v34 = *v33;
      if ( v9 == *v33 )
      {
LABEL_37:
        v11 = (unsigned __int64)v33[1];
        if ( v11 )
          goto LABEL_6;
      }
      else
      {
        v36 = 1;
        while ( v34 != (unsigned __int8 *)-4096LL )
        {
          v37 = v36 + 1;
          v32 = v31 & (v36 + v32);
          v33 = (unsigned __int8 **)(v30 + 16LL * v32);
          v34 = *v33;
          if ( v9 == *v33 )
            goto LABEL_37;
          v36 = v37;
        }
      }
    }
    v11 = (unsigned __int64)sub_3250680(a1, v9, 0);
  }
  else if ( (unsigned __int8)v10 <= 0x24u && (v35 = 0x140000F000LL, _bittest64(&v35, v10)) )
  {
    v11 = (unsigned __int64)sub_3251AD0(a1, v9);
  }
  else if ( (_BYTE)v10 == 25 )
  {
    v11 = sub_373FFB0(a1, (__int64)v9, 0, 0);
  }
  else if ( (_BYTE)v10 == 29 )
  {
    v11 = (unsigned __int64)sub_37409E0(a1, v9);
  }
  else
  {
    v11 = (unsigned __int64)sub_3247C80((__int64)a1, v9);
  }
LABEL_6:
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v7 - 8LL * ((v12 >> 2) & 0xF);
  sub_3249CA0(a1, v6, *(_DWORD *)(a2 + 4), *(_QWORD *)(v13 + 24));
  sub_32494F0(a1, v6, 24, v11);
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( !v15 )
      goto LABEL_24;
    goto LABEL_10;
  }
  v15 = *(_QWORD *)(a2 - 8LL * ((v14 >> 2) & 0xF));
  if ( v15 )
  {
LABEL_10:
    v16 = (const void *)sub_B91420(v15);
    v18 = v17;
    if ( v17 )
    {
      v38 = (__int64)v16;
      sub_324AD70(a1, v6, 3, v16, v17);
      sub_32382C0(a1[26], (__int64)a1, *(_DWORD *)(a1[10] + 36), v38, v18, v6);
    }
    v14 = *(_BYTE *)(a2 - 16);
    if ( (v14 & 2) == 0 )
      goto LABEL_13;
LABEL_24:
    v19 = *(_QWORD *)(a2 - 32);
    goto LABEL_14;
  }
LABEL_13:
  v19 = v7 - 8LL * ((v14 >> 2) & 0xF);
LABEL_14:
  v20 = *(_QWORD *)(v19 + 32);
  if ( v20 )
  {
    v21 = *(_BYTE *)(v20 - 16);
    if ( (v21 & 2) != 0 )
    {
      v22 = *(_QWORD **)(v20 - 32);
      v23 = *(unsigned int *)(v20 - 24);
    }
    else
    {
      v22 = (_QWORD *)(v20 - 16 - 8LL * ((v21 >> 2) & 0xF));
      v23 = (*(_WORD *)(v20 - 16) >> 6) & 0xF;
    }
    for ( i = &v22[v23]; i != v22; ++v22 )
    {
      if ( *v22 )
      {
        v25 = sub_3740A90(a1, *v22);
        *(_QWORD *)(v25 + 40) = v6 & 0xFFFFFFFFFFFFFFFBLL;
        v26 = *(_QWORD **)(v6 + 32);
        if ( v26 )
        {
          *(_QWORD *)v25 = *v26;
          **(_QWORD **)(v6 + 32) = v25 & 0xFFFFFFFFFFFFFFFBLL;
        }
        *(_QWORD *)(v6 + 32) = v25;
      }
    }
  }
  return v6;
}
