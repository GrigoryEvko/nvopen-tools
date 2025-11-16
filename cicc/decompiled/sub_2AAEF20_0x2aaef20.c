// Function: sub_2AAEF20
// Address: 0x2aaef20
//
__int64 __fastcall sub_2AAEF20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // eax
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *j; // rdx
  __int64 v9; // rax
  unsigned __int64 *v10; // rdx
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 *v15; // rcx
  unsigned __int64 v16; // r13
  __int64 *v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  int v22; // r12d
  _QWORD *v23; // rax
  unsigned int v24; // eax
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *i; // rdx
  __int64 *v28; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 *v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  _QWORD v32[8]; // [rsp+20h] [rbp-40h] BYREF

  result = 0;
  if ( *(_DWORD *)(a1 + 104) != *(_DWORD *)(a1 + 100) )
  {
    v4 = *(_DWORD *)(a1 + 64);
    ++*(_QWORD *)(a1 + 48);
    if ( v4 )
    {
      v5 = 4 * v4;
      a2 = 64;
      v6 = *(unsigned int *)(a1 + 72);
      if ( (unsigned int)(4 * v4) < 0x40 )
        v5 = 64;
      if ( (unsigned int)v6 <= v5 )
        goto LABEL_6;
      v20 = v4 - 1;
      if ( v20 )
      {
        _BitScanReverse(&v20, v20);
        v21 = *(_QWORD **)(a1 + 56);
        v22 = 1 << (33 - (v20 ^ 0x1F));
        if ( v22 < 64 )
          v22 = 64;
        if ( (_DWORD)v6 == v22 )
        {
          *(_QWORD *)(a1 + 64) = 0;
          v23 = &v21[2 * (unsigned int)v6];
          do
          {
            if ( v21 )
              *v21 = -4096;
            v21 += 2;
          }
          while ( v23 != v21 );
          goto LABEL_9;
        }
      }
      else
      {
        v21 = *(_QWORD **)(a1 + 56);
        v22 = 64;
      }
      a2 = 16 * v6;
      sub_C7D6A0((__int64)v21, 16 * v6, 8);
      v24 = sub_2AAAC60(v22);
      *(_DWORD *)(a1 + 72) = v24;
      if ( v24 )
      {
        a2 = 8;
        v25 = (_QWORD *)sub_C7D670(16LL * v24, 8);
        v26 = *(unsigned int *)(a1 + 72);
        *(_QWORD *)(a1 + 64) = 0;
        *(_QWORD *)(a1 + 56) = v25;
        for ( i = &v25[2 * v26]; i != v25; v25 += 2 )
        {
          if ( v25 )
            *v25 = -4096;
        }
LABEL_9:
        v9 = *(_QWORD *)(a1 + 88);
        if ( *(_BYTE *)(a1 + 108) )
          v10 = (unsigned __int64 *)(v9 + 8LL * *(unsigned int *)(a1 + 100));
        else
          v10 = (unsigned __int64 *)(v9 + 8LL * *(unsigned int *)(a1 + 96));
        v28 = *(__int64 **)(a1 + 88);
        v29 = v10;
        sub_254BBF0((__int64)&v28);
        v11 = *(_QWORD *)(a1 + 80);
        v12 = *(_BYTE *)(a1 + 108) == 0;
        v30 = a1 + 80;
        v31 = v11;
        if ( v12 )
          v13 = *(unsigned int *)(a1 + 96);
        else
          v13 = *(unsigned int *)(a1 + 100);
        v32[0] = *(_QWORD *)(a1 + 88) + 8 * v13;
        v32[1] = v32[0];
        sub_254BBF0((__int64)v32);
        v14 = *(_QWORD *)(a1 + 80);
        v32[2] = a1 + 80;
        v15 = (unsigned __int64 *)v28;
        v32[3] = v14;
        if ( v28 != (__int64 *)v32[0] )
        {
          do
          {
            while ( 1 )
            {
              v16 = *v15;
              if ( *v15 )
              {
                sub_C7D6A0(*(_QWORD *)(v16 + 16), 16LL * *(unsigned int *)(v16 + 32), 8);
                a2 = 56;
                j_j___libc_free_0(v16);
              }
              v15 = v29;
              v17 = v28 + 1;
              v28 = v17;
              if ( v17 != (__int64 *)v29 )
                break;
LABEL_19:
              if ( (unsigned __int64 *)v32[0] == v29 )
                goto LABEL_20;
            }
            while ( 1 )
            {
              a2 = *v17;
              if ( (unsigned __int64)(*v17 + 2) > 1 )
                break;
              v28 = ++v17;
              if ( v17 == (__int64 *)v29 )
                goto LABEL_19;
            }
            v15 = (unsigned __int64 *)v28;
          }
          while ( (__int64 *)v32[0] != v28 );
LABEL_20:
          v14 = *(_QWORD *)(a1 + 80);
        }
        v12 = *(_BYTE *)(a1 + 108) == 0;
        *(_QWORD *)(a1 + 80) = v14 + 1;
        if ( v12 )
        {
          v18 = 4 * (*(_DWORD *)(a1 + 100) - *(_DWORD *)(a1 + 104));
          v19 = *(unsigned int *)(a1 + 96);
          if ( v18 < 0x20 )
            v18 = 32;
          if ( (unsigned int)v19 > v18 )
          {
            sub_C8C990(a1 + 80, a2);
            goto LABEL_27;
          }
          memset(*(void **)(a1 + 88), -1, 8 * v19);
        }
        *(_QWORD *)(a1 + 100) = 0;
LABEL_27:
        *(_BYTE *)(a1 + 40) = 0;
        return 1;
      }
    }
    else
    {
      if ( !*(_DWORD *)(a1 + 68) )
        goto LABEL_9;
      v6 = *(unsigned int *)(a1 + 72);
      if ( (unsigned int)v6 <= 0x40 )
      {
LABEL_6:
        v7 = *(_QWORD **)(a1 + 56);
        for ( j = &v7[2 * v6]; j != v7; v7 += 2 )
          *v7 = -4096;
        goto LABEL_8;
      }
      a2 = 16 * v6;
      sub_C7D6A0(*(_QWORD *)(a1 + 56), 16 * v6, 8);
      *(_DWORD *)(a1 + 72) = 0;
    }
    *(_QWORD *)(a1 + 56) = 0;
LABEL_8:
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_9;
  }
  return result;
}
