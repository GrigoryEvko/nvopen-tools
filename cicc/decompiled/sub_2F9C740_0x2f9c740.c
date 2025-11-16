// Function: sub_2F9C740
// Address: 0x2f9c740
//
unsigned __int64 __fastcall sub_2F9C740(unsigned __int8 **a1, char a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  char v7; // al
  int v8; // esi
  _BYTE *v9; // r8
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // edx
  _QWORD *v13; // rcx
  _BYTE *v14; // rdi
  unsigned __int64 result; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  int v19; // edx
  __int64 v20; // rcx
  _BYTE *v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r9
  unsigned int v24; // ecx
  __int64 v25; // r8
  _BYTE *v26; // r10
  unsigned __int64 v27; // rdx
  unsigned __int16 v28; // ax
  __int16 v29; // cx
  int v30; // ecx
  int v31; // r10d
  int v32; // r8d
  int v33; // ebx
  unsigned __int16 v34; // [rsp+Ch] [rbp-24h] BYREF
  unsigned __int16 v35; // [rsp+Eh] [rbp-22h] BYREF
  unsigned __int64 v36; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v37[3]; // [rsp+18h] [rbp-18h] BYREF

  v6 = (__int64)*a1;
  v7 = *((_BYTE *)a1 + 8);
  v8 = **a1;
  if ( a2 )
  {
    if ( v7 )
      goto LABEL_3;
LABEL_12:
    if ( (_BYTE)v8 == 86 )
    {
      v9 = *(_BYTE **)(v6 - 64);
      goto LABEL_5;
    }
    if ( (unsigned int)(v8 - 42) <= 0x11 )
      goto LABEL_15;
LABEL_43:
    BUG();
  }
  if ( v7 )
    goto LABEL_12;
LABEL_3:
  if ( (_BYTE)v8 == 86 )
  {
    v9 = *(_BYTE **)(v6 - 32);
LABEL_5:
    if ( v9 )
    {
      if ( *v9 > 0x1Cu )
        goto LABEL_7;
      return 0;
    }
    v8 = 86;
    goto LABEL_15;
  }
  if ( (unsigned int)(v8 - 42) > 0x11 )
    goto LABEL_43;
  v9 = *(_BYTE **)(v6 + 32LL * (unsigned int)(1 - *((_DWORD *)a1 + 3)) - 64);
  if ( v9 )
  {
    if ( *v9 > 0x1Cu )
    {
LABEL_7:
      v10 = *(unsigned int *)(a3 + 24);
      v11 = *(_QWORD *)(a3 + 8);
      if ( (_DWORD)v10 )
      {
        v12 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v13 = (_QWORD *)(v11 + 40LL * v12);
        v14 = (_BYTE *)*v13;
        if ( (_BYTE *)*v13 == v9 )
        {
LABEL_9:
          if ( v13 != (_QWORD *)(v11 + 40 * v10) )
            return v13[3];
        }
        else
        {
          v30 = 1;
          while ( v14 != (_BYTE *)-4096LL )
          {
            v31 = v30 + 1;
            v12 = (v10 - 1) & (v30 + v12);
            v13 = (_QWORD *)(v11 + 40LL * v12);
            v14 = (_BYTE *)*v13;
            if ( (_BYTE *)*v13 == v9 )
              goto LABEL_9;
            v30 = v31;
          }
        }
      }
    }
    return 0;
  }
LABEL_15:
  v16 = sub_DFD800(a4, v8 - 29, *(_QWORD *)(v6 + 8), 1u, 0, 0x100000002LL, 0, 0, 0, 0);
  v17 = 0;
  v18 = (__int64)*a1;
  if ( !v19 )
    v17 = v16;
  result = v17;
  if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
    v20 = *(_QWORD *)(v18 - 8);
  else
    v20 = v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
  v21 = *(_BYTE **)(v20 + 32LL * (unsigned int)(1 - *((_DWORD *)a1 + 3)));
  if ( *v21 > 0x1Cu )
  {
    v22 = *(unsigned int *)(a3 + 24);
    v23 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v22 )
    {
      v24 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v25 = v23 + 40LL * v24;
      v26 = *(_BYTE **)v25;
      if ( *(_BYTE **)v25 == v21 )
      {
LABEL_22:
        if ( v25 != v23 + 40 * v22 )
        {
          v27 = *(_QWORD *)(v25 + 24);
          v28 = *(_WORD *)(v25 + 32);
          v36 = v17;
          v34 = 0;
          v37[0] = v27;
          v35 = v28;
          v29 = sub_FDCA70(&v36, &v34, v37, &v35);
          result = v36 + v37[0];
          if ( __CFADD__(v36, v37[0]) )
          {
            ++v29;
            result = (result >> 1) | 0x8000000000000000LL;
          }
          if ( v29 > 0x3FFF )
            return -1;
        }
      }
      else
      {
        v32 = 1;
        while ( v26 != (_BYTE *)-4096LL )
        {
          v33 = v32 + 1;
          v24 = (v22 - 1) & (v32 + v24);
          v25 = v23 + 40LL * v24;
          v26 = *(_BYTE **)v25;
          if ( v21 == *(_BYTE **)v25 )
            goto LABEL_22;
          v32 = v33;
        }
      }
    }
  }
  return result;
}
