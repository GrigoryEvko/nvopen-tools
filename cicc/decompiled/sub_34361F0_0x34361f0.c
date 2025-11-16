// Function: sub_34361F0
// Address: 0x34361f0
//
unsigned __int64 __fastcall sub_34361F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // r12
  int v6; // r15d
  unsigned __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // r8d
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  _DWORD *v19; // rax
  _DWORD *v20; // rcx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rdi
  unsigned int v26; // esi
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int); // rax
  int v28; // edx
  unsigned __int16 v29; // ax
  _QWORD *v30; // r14
  int v31; // edx
  int v32; // r15d
  unsigned __int64 *v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+28h] [rbp-48h]
  _QWORD v37[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = sub_338B750(a2, a1);
  v5 = v4;
  v6 = v4;
  result = sub_3433840(v3, v4, v4);
  if ( (_BYTE)result )
    return result;
  v8 = *(unsigned int *)(a2 + 296);
  v9 = *(_QWORD *)(a2 + 280);
  if ( (_DWORD)v8 )
  {
    v10 = 1;
    for ( result = ((_DWORD)v8 - 1) & ((unsigned int)v5 + ((unsigned int)(v3 >> 9) ^ (unsigned int)(v3 >> 4)));
          ;
          result = ((_DWORD)v8 - 1) & v12 )
    {
      v11 = v9 + 32LL * (unsigned int)result;
      if ( v3 == *(_QWORD *)v11 && v6 == *(_DWORD *)(v11 + 8) )
        break;
      if ( !*(_QWORD *)v11 && *(_DWORD *)(v11 + 8) == -1 )
        goto LABEL_9;
      v12 = v10 + result;
      ++v10;
    }
    if ( v11 != v9 + 32 * v8 && *(_QWORD *)(v11 + 16) )
      return result;
  }
LABEL_9:
  result = sub_3435EB0(a1, a2, 6);
  v36 = result;
  if ( !BYTE4(result) )
    return result;
  v13 = *(_QWORD *)(a2 + 960);
  v14 = *(_QWORD *)(v13 + 528);
  v15 = 4LL * *(unsigned int *)(v13 + 536);
  v16 = v14 + v15;
  v17 = v15 >> 2;
  v18 = v15 >> 4;
  if ( v18 )
  {
    v19 = *(_DWORD **)(v13 + 528);
    v20 = (_DWORD *)(v14 + 16 * v18);
    while ( *v19 != v36 )
    {
      if ( v19[1] == v36 )
      {
        v17 = ((__int64)v19 - v14 + 4) >> 2;
        goto LABEL_18;
      }
      if ( v19[2] == v36 )
      {
        v17 = ((__int64)v19 - v14 + 8) >> 2;
        goto LABEL_18;
      }
      if ( v19[3] == v36 )
      {
        v17 = ((__int64)v19 - v14 + 12) >> 2;
        goto LABEL_18;
      }
      v19 += 4;
      if ( v20 == v19 )
      {
        v34 = (v16 - (__int64)v19) >> 2;
        goto LABEL_41;
      }
    }
    goto LABEL_17;
  }
  v34 = v17;
  v19 = *(_DWORD **)(v13 + 528);
LABEL_41:
  if ( v34 == 2 )
  {
LABEL_49:
    if ( *v19 != v36 )
    {
      ++v19;
LABEL_44:
      if ( *v19 != v36 )
        goto LABEL_18;
      goto LABEL_17;
    }
    goto LABEL_17;
  }
  if ( v34 != 3 )
  {
    if ( v34 != 1 )
      goto LABEL_18;
    goto LABEL_44;
  }
  if ( *v19 != v36 )
  {
    ++v19;
    goto LABEL_49;
  }
LABEL_17:
  v17 = ((__int64)v19 - v14) >> 2;
LABEL_18:
  v21 = *(_QWORD *)(a2 + 304);
  if ( (v21 & 1) != 0 )
  {
    v22 = v21 >> 58;
    result = ~(-1LL << (v21 >> 58));
    v23 = result & (v21 >> 1);
    if ( _bittest64((const __int64 *)&v23, v17) )
      return result;
    *(_QWORD *)(a2 + 304) = 2 * ((v22 << 57) | result & (v23 | (1LL << v17))) + 1;
  }
  else
  {
    v33 = (unsigned __int64 *)(*(_QWORD *)v21 + 8LL * ((unsigned int)v17 >> 6));
    result = *v33;
    if ( _bittest64((const __int64 *)&result, v17) )
      return result;
    *v33 = result | (1LL << (v17 & 0x3F));
  }
  v24 = *(_QWORD *)(a2 + 864);
  v35 = *(_QWORD *)(v24 + 16);
  v25 = sub_2E79000(*(__int64 **)(v24 + 40));
  v26 = *(_DWORD *)(v25 + 4);
  v27 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v35 + 32LL);
  if ( v27 == sub_2D42F30 )
  {
    v28 = sub_AE2980(v25, v26)[1];
    v29 = 2;
    if ( v28 != 1 )
    {
      v29 = 3;
      if ( v28 != 2 )
      {
        v29 = 4;
        if ( v28 != 4 )
        {
          v29 = 5;
          if ( v28 != 8 )
          {
            v29 = 6;
            if ( v28 != 16 )
            {
              v29 = 7;
              if ( v28 != 32 )
              {
                v29 = 8;
                if ( v28 != 64 )
                  v29 = 9 * (v28 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v29 = v27(v35, v25, v26);
  }
  v37[0] = v3;
  v30 = sub_33EDBD0((_QWORD *)v24, v36, v29, 0, 1);
  v32 = v31;
  v37[1] = v5;
  result = (unsigned __int64)sub_34348A0(a2 + 272, (__int64)v37);
  *(_QWORD *)result = v30;
  *(_DWORD *)(result + 8) = v32;
  return result;
}
