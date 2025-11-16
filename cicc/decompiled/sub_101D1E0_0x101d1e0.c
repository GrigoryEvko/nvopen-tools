// Function: sub_101D1E0
// Address: 0x101d1e0
//
unsigned __int8 *__fastcall sub_101D1E0(__int64 a1, _BYTE *a2, char a3, char a4, __m128i *a5, int a6)
{
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned __int8 *result; // rax
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  unsigned __int8 v16; // cl
  __int64 v17; // r14
  _BYTE *v18; // rax
  char v19; // dl
  unsigned int v20; // edi
  __int64 v21; // rdx
  int v22; // r12d
  unsigned int v23; // ebx
  __int64 v24; // rax
  unsigned int v25; // edi
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rax
  int v29; // r14d
  unsigned int v30; // r15d
  __int64 v31; // rax
  unsigned int v32; // r8d
  __int64 v33; // rdx
  char v34; // [rsp+7h] [rbp-39h]
  char v35; // [rsp+7h] [rbp-39h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v8 = (__int64)a2;
  v9 = a1;
  result = sub_101CD30(0x19u, (_BYTE *)a1, a2, a3, a5, a6);
  if ( !result )
  {
    v12 = a1;
    v36 = *(_QWORD *)(a1 + 8);
    v13 = sub_1003090((__int64)a5, (unsigned __int8 *)a1);
    v14 = v13;
    if ( (_BYTE)v13 )
    {
      if ( a3 || a4 )
        return (unsigned __int8 *)v9;
      return (unsigned __int8 *)sub_AD6530(v36, v12);
    }
    if ( !a5[4].m128i_i8[0] )
    {
LABEL_4:
      if ( !a4 )
        return 0;
      v16 = *(_BYTE *)a1;
      if ( *(_BYTE *)a1 == 17 )
      {
        v25 = *(_DWORD *)(a1 + 32);
        v12 = *(_QWORD *)(v9 + 24);
        v21 = 1LL << ((unsigned __int8)v25 - 1);
        if ( v25 > 0x40 )
          v12 = *(_QWORD *)(v12 + 8LL * ((v25 - 1) >> 6));
        goto LABEL_22;
      }
LABEL_16:
      v17 = *(_QWORD *)(a1 + 8);
      v34 = v14;
      if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 > 1 || v16 > 0x15u )
        goto LABEL_23;
      v12 = 0;
      v18 = sub_AD7630(a1, 0, v14);
      v19 = v34;
      if ( !v18 || *v18 != 17 )
      {
        if ( *(_BYTE *)(v17 + 8) == 17 )
        {
          v29 = *(_DWORD *)(v17 + 32);
          if ( v29 )
          {
            v30 = 0;
            while ( 1 )
            {
              v12 = v30;
              v35 = v19;
              v31 = sub_AD69F0((unsigned __int8 *)a1, v30);
              if ( !v31 )
                break;
              v19 = v35;
              if ( *(_BYTE *)v31 != 13 )
              {
                if ( *(_BYTE *)v31 != 17 )
                  goto LABEL_23;
                v32 = *(_DWORD *)(v31 + 32);
                v33 = *(_QWORD *)(v31 + 24);
                if ( v32 > 0x40 )
                  v33 = *(_QWORD *)(v33 + 8LL * ((v32 - 1) >> 6));
                if ( (v33 & (1LL << ((unsigned __int8)v32 - 1))) == 0 )
                  goto LABEL_23;
                v19 = 1;
              }
              if ( v29 == ++v30 )
              {
                if ( v19 )
                  return (unsigned __int8 *)v9;
                goto LABEL_23;
              }
            }
          }
        }
        goto LABEL_23;
      }
      v20 = *((_DWORD *)v18 + 8);
      v12 = *((_QWORD *)v18 + 3);
      v21 = 1LL << ((unsigned __int8)v20 - 1);
      if ( v20 > 0x40 )
        v12 = *(_QWORD *)(v12 + 8LL * ((v20 - 1) >> 6));
LABEL_22:
      if ( (v12 & v21) != 0 )
        return (unsigned __int8 *)v9;
LABEL_23:
      if ( !a3 )
        return 0;
      v22 = sub_BCB060(v36);
      if ( *(_BYTE *)v8 != 17 )
      {
        v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
        if ( (unsigned int)v26 > 1 )
          return 0;
        if ( *(_BYTE *)v8 > 0x15u )
          return 0;
        v12 = 0;
        v27 = sub_AD7630(v8, 0, v26);
        v8 = (__int64)v27;
        if ( !v27 || *v27 != 17 )
          return 0;
      }
      v23 = *(_DWORD *)(v8 + 32);
      if ( v23 > 0x40 )
      {
        if ( v23 - (unsigned int)sub_C444A0(v8 + 24) > 0x40 )
          return 0;
        v24 = **(_QWORD **)(v8 + 24);
      }
      else
      {
        v24 = *(_QWORD *)(v8 + 24);
      }
      if ( v22 - 1 != v24 )
        return 0;
      return (unsigned __int8 *)sub_AD6530(v36, v12);
    }
    v15 = *(_BYTE *)a1;
    v16 = *(_BYTE *)a1;
    if ( *(_BYTE *)a1 <= 0x1Cu )
    {
      if ( v15 != 5 )
        goto LABEL_4;
      v12 = *(unsigned __int16 *)(a1 + 2);
      if ( (unsigned int)(v12 - 19) <= 1 || (v12 = (unsigned int)(v12 - 26), (unsigned __int16)v12 <= 1u) )
      {
LABEL_14:
        if ( !a4 )
          return 0;
        v16 = *(_BYTE *)a1;
        goto LABEL_16;
      }
    }
    else
    {
      v12 = v15;
      if ( (unsigned int)v15 - 48 <= 1 || (unsigned __int8)(v15 - 55) <= 1u )
      {
        if ( (*(_BYTE *)(a1 + 1) & 2) == 0 )
          goto LABEL_14;
        v12 = (unsigned int)v15 - 55;
        if ( (unsigned int)v12 <= 1 )
        {
          if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          {
            v12 = *(_QWORD *)(a1 - 8);
            v28 = *(_QWORD *)v12;
            if ( !*(_QWORD *)v12 )
              goto LABEL_35;
          }
          else
          {
            v12 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
            v28 = *(_QWORD *)v12;
            if ( !*(_QWORD *)v12 )
              goto LABEL_35;
          }
          if ( v8 == *(_QWORD *)(v12 + 32) )
            return (unsigned __int8 *)v28;
        }
      }
    }
LABEL_35:
    if ( !a4 )
      return 0;
    goto LABEL_16;
  }
  return result;
}
