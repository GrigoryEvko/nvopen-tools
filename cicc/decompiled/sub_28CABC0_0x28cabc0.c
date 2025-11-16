// Function: sub_28CABC0
// Address: 0x28cabc0
//
unsigned __int8 **__fastcall sub_28CABC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // rax
  unsigned __int8 **v7; // rdx
  unsigned __int8 **v8; // rdi
  unsigned __int8 **v9; // rdx
  unsigned __int8 **result; // rax
  unsigned __int64 v11; // r9
  __int64 v12; // r10
  int v13; // esi
  __int64 v14; // r9
  int v15; // esi
  unsigned int v16; // r11d
  __int64 *v17; // rcx
  __int64 v18; // r12
  unsigned int v19; // ecx
  __int64 v20; // r9
  __int64 v21; // rsi
  int v22; // esi
  unsigned int v23; // r11d
  __int64 v24; // r12
  int v25; // ecx
  int v26; // ecx
  int v27; // r13d
  int v28; // r13d
  unsigned __int8 **v29; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int8 **v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  _QWORD v33[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a2 + 156) )
    v3 = *(unsigned int *)(a2 + 148);
  else
    v3 = *(unsigned int *)(a2 + 144);
  v33[0] = *(_QWORD *)(a2 + 136) + 8 * v3;
  v33[1] = v33[0];
  sub_254BBF0((__int64)v33);
  v4 = *(_QWORD *)(a2 + 128);
  v33[2] = a2 + 128;
  v5 = *(_BYTE *)(a2 + 156) == 0;
  v33[3] = v4;
  v6 = *(_QWORD *)(a2 + 136);
  if ( v5 )
    v7 = (unsigned __int8 **)(v6 + 8LL * *(unsigned int *)(a2 + 144));
  else
    v7 = (unsigned __int8 **)(v6 + 8LL * *(unsigned int *)(a2 + 148));
  v29 = *(unsigned __int8 ***)(a2 + 136);
  v30 = v7;
  sub_254BBF0((__int64)&v29);
  v8 = (unsigned __int8 **)v33[0];
  v31 = a2 + 128;
  v9 = v30;
  v32 = *(_QWORD *)(a2 + 128);
  result = v29;
  while ( v8 != result )
  {
    v11 = (unsigned __int64)*result;
    v12 = *(_QWORD *)(a1 + 2424);
    v13 = *(_DWORD *)(a1 + 2440);
    if ( (unsigned int)**result - 26 > 1 )
    {
      if ( !v13 )
        goto LABEL_19;
      v22 = v13 - 1;
      v23 = v22 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v17 = (__int64 *)(v12 + 16LL * v23);
      v24 = *v17;
      if ( v11 != *v17 )
      {
        v25 = 1;
        while ( v24 != -4096 )
        {
          v28 = v25 + 1;
          v23 = v22 & (v25 + v23);
          v17 = (__int64 *)(v12 + 16LL * v23);
          v24 = *v17;
          if ( v11 == *v17 )
            goto LABEL_9;
          v25 = v28;
        }
        goto LABEL_19;
      }
    }
    else
    {
      v14 = *(_QWORD *)(v11 + 72);
      if ( !v13 )
        goto LABEL_19;
      v15 = v13 - 1;
      v16 = v15 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v17 = (__int64 *)(v12 + 16LL * v16);
      v18 = *v17;
      if ( *v17 != v14 )
      {
        v26 = 1;
        while ( v18 != -4096 )
        {
          v27 = v26 + 1;
          v16 = v15 & (v26 + v16);
          v17 = (__int64 *)(v12 + 16LL * v16);
          v18 = *v17;
          if ( v14 == *v17 )
            goto LABEL_9;
          v26 = v27;
        }
LABEL_19:
        v20 = 1;
        v21 = 0;
        goto LABEL_10;
      }
    }
LABEL_9:
    v19 = *((_DWORD *)v17 + 2);
    v20 = 1LL << v19;
    v21 = 8LL * (v19 >> 6);
LABEL_10:
    ++result;
    for ( *(_QWORD *)(*(_QWORD *)(a1 + 2280) + v21) |= v20; v9 != result; ++result )
    {
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
  }
  return result;
}
