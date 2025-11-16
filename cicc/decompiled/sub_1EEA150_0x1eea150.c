// Function: sub_1EEA150
// Address: 0x1eea150
//
__int64 __fastcall sub_1EEA150(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int8 *v10; // r13
  __int64 result; // rax
  unsigned __int8 *v12; // r12
  signed int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rsi
  unsigned int v17; // edi
  unsigned __int16 *v18; // rax
  int v19; // ecx
  int v20; // edx
  int v21; // eax
  unsigned int v22; // esi
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  _QWORD *v28; // [rsp+8h] [rbp-38h]

  v6 = a1 + 16;
  v7 = a1[17];
  v8 = a1[4];
  if ( v7 )
    memset((void *)a1[16], 0, 8 * v7);
  v28 = a1 + 19;
  v9 = a1[20];
  if ( v9 )
    memset((void *)a1[19], 0, 8 * v9);
  v10 = *(unsigned __int8 **)(v8 + 32);
  result = 5LL * *(unsigned int *)(v8 + 40);
  v12 = &v10[40 * *(unsigned int *)(v8 + 40)];
  if ( v12 != v10 )
  {
    while ( 1 )
    {
      result = *v10;
      if ( (_BYTE)result == 12 )
      {
        *((_DWORD *)a1 + 48) = 0;
        v14 = *a1;
        v15 = *(unsigned int *)(*a1 + 44);
        if ( (_DWORD)v15 )
        {
          v16 = 0;
LABEL_21:
          v17 = v16;
          if ( !v14 )
            BUG();
          v18 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 4 * v16);
          v19 = *v18;
          v20 = v18[1];
          do
          {
            if ( !(_WORD)v19 )
            {
              if ( v15 == ++v16 )
                goto LABEL_28;
LABEL_26:
              v14 = *a1;
              goto LABEL_21;
            }
            v21 = *(_DWORD *)(*((_QWORD *)v10 + 3) + 4 * ((unsigned __int64)(unsigned __int16)v19 >> 5)) >> v19;
            v19 = v20;
            v20 = 0;
          }
          while ( (v21 & 1) != 0 );
          v19 = v16++;
          a5 = 1LL << v17;
          *(_QWORD *)(a1[22] + 8LL * (v17 >> 6)) |= 1LL << v17;
          if ( v15 != v16 )
            goto LABEL_26;
LABEL_28:
          v22 = *((_DWORD *)a1 + 48);
          if ( *((_DWORD *)a1 + 36) < v22 )
          {
            sub_13A49F0((__int64)v6, v22, 0, v19, a5, v15);
            v23 = (unsigned int)(*((_DWORD *)a1 + 48) + 63) >> 6;
          }
          else
          {
            v23 = (v22 + 63) >> 6;
          }
          v24 = 8 * v23;
          v25 = 0;
          if ( v23 )
          {
            do
            {
              v26 = (_QWORD *)(v25 + a1[16]);
              v27 = *(_QWORD *)(a1[22] + v25);
              v25 += 8;
              *v26 |= v27;
            }
            while ( v24 != v25 );
          }
        }
        result = *v10;
      }
      if ( (_BYTE)result )
        goto LABEL_10;
      v13 = *((_DWORD *)v10 + 2);
      if ( v13 <= 0 )
        goto LABEL_10;
      result = *(_QWORD *)(*(_QWORD *)(a1[2] + 304) + 8LL * ((unsigned int)v13 >> 6)) & (1LL << v13);
      if ( result )
        goto LABEL_10;
      result = v10[3];
      if ( (result & 0x10) == 0 )
        break;
      if ( (((result & 0x10) != 0) & ((unsigned __int8)result >> 6)) != 0 )
      {
        v10 += 40;
        result = sub_1EEA0C0(a1, v6, v13);
        if ( v12 == v10 )
          return result;
      }
      else
      {
        result = sub_1EEA0C0(a1, v28, v13);
LABEL_10:
        v10 += 40;
        if ( v12 == v10 )
          return result;
      }
    }
    if ( (v10[4] & 1) == 0 && (result & 0x40) != 0 )
      result = sub_1EEA0C0(a1, v6, v13);
    goto LABEL_10;
  }
  return result;
}
