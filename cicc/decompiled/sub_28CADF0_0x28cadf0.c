// Function: sub_28CADF0
// Address: 0x28cadf0
//
_BYTE **__fastcall sub_28CADF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _BYTE **v4; // rdx
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rdx
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  _BYTE **result; // rax
  _BYTE *v14; // rsi
  int v15; // eax
  int v16; // edi
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rax
  int v21; // eax
  _BYTE **v22; // [rsp+0h] [rbp-60h] BYREF
  _BYTE **v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  _QWORD v26[8]; // [rsp+20h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  if ( *(_BYTE *)(a2 + 92) )
    v4 = (_BYTE **)(v3 + 8LL * *(unsigned int *)(a2 + 84));
  else
    v4 = (_BYTE **)(v3 + 8LL * *(unsigned int *)(a2 + 80));
  v22 = *(_BYTE ***)(a2 + 72);
  v23 = v4;
  sub_254BBF0((__int64)&v22);
  v5 = *(_QWORD *)(a2 + 64);
  v6 = *(_BYTE *)(a2 + 92) == 0;
  v24 = a2 + 64;
  v25 = v5;
  if ( v6 )
    v7 = *(unsigned int *)(a2 + 80);
  else
    v7 = *(unsigned int *)(a2 + 84);
  v26[0] = *(_QWORD *)(a2 + 72) + 8 * v7;
  v26[1] = v26[0];
  sub_254BBF0((__int64)v26);
  v12 = *(_QWORD *)(a2 + 64);
  v26[2] = a2 + 64;
  v26[3] = v12;
  result = v22;
  if ( v22 != (_BYTE **)v26[0] )
  {
    while ( 1 )
    {
      v14 = *result;
      if ( **result > 0x1Cu )
        break;
LABEL_11:
      if ( !*(_BYTE *)(a1 + 2084) )
        goto LABEL_21;
      v20 = *(_QWORD **)(a1 + 2064);
      v9 = *(unsigned int *)(a1 + 2076);
      v8 = (unsigned __int64)&v20[v9];
      if ( v20 != (_QWORD *)v8 )
      {
        while ( v14 != (_BYTE *)*v20 )
        {
          if ( (_QWORD *)v8 == ++v20 )
            goto LABEL_24;
        }
        goto LABEL_16;
      }
LABEL_24:
      if ( (unsigned int)v9 < *(_DWORD *)(a1 + 2072) )
      {
        *(_DWORD *)(a1 + 2076) = v9 + 1;
        *(_QWORD *)v8 = v14;
        ++*(_QWORD *)(a1 + 2056);
      }
      else
      {
LABEL_21:
        sub_C8CC70(a1 + 2056, (__int64)v14, v8, v9, v10, v11);
      }
LABEL_16:
      v9 = (__int64)v23;
      result = v22 + 1;
      v22 = result;
      if ( result == v23 )
      {
LABEL_19:
        if ( (_BYTE **)v26[0] == result )
          return result;
      }
      else
      {
        while ( 1 )
        {
          v8 = (unsigned __int64)(*result + 2);
          if ( v8 > 1 )
            break;
          v22 = ++result;
          if ( v23 == result )
            goto LABEL_19;
        }
        result = v22;
        if ( (_BYTE **)v26[0] == v22 )
          return result;
      }
    }
    v15 = *(_DWORD *)(a1 + 2440);
    v9 = *(_QWORD *)(a1 + 2424);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = v9 + 16LL * v17;
      v10 = *(_QWORD *)v18;
      if ( v14 == *(_BYTE **)v18 )
      {
LABEL_9:
        v9 = *(unsigned int *)(v18 + 8);
        v8 = 1LL << v9;
        v19 = 8LL * ((unsigned int)v9 >> 6);
LABEL_10:
        *(_QWORD *)(*(_QWORD *)(a1 + 2280) + v19) |= v8;
        goto LABEL_11;
      }
      v21 = 1;
      while ( v10 != -4096 )
      {
        v11 = (unsigned int)(v21 + 1);
        v17 = v16 & (v21 + v17);
        v18 = v9 + 16LL * v17;
        v10 = *(_QWORD *)v18;
        if ( v14 == *(_BYTE **)v18 )
          goto LABEL_9;
        v21 = v11;
      }
    }
    v8 = 1;
    v19 = 0;
    goto LABEL_10;
  }
  return result;
}
