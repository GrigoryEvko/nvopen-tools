// Function: sub_1BF1A00
// Address: 0x1bf1a00
//
void __fastcall sub_1BF1A00(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r9d
  unsigned int v3; // r14d
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 *v6; // r13
  __int64 v7; // r14
  unsigned int v8; // r10d
  __int64 v9; // r8
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  _BYTE *v12; // r11
  __int64 *v13; // rcx
  __int64 *v14; // rsi
  __int64 v15; // rbx
  _BYTE *v16; // r15
  __int64 v17; // rdx
  unsigned int v18; // r14d
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // [rsp+18h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  __int64 v23; // [rsp+28h] [rbp-68h]
  __int64 *v24; // [rsp+30h] [rbp-60h] BYREF
  __int64 v25; // [rsp+38h] [rbp-58h]
  char v26; // [rsp+40h] [rbp-50h] BYREF

  v1 = sub_13FD000(*(_QWORD *)(a1 + 72));
  if ( v1 )
  {
    v3 = *(_DWORD *)(v1 + 8);
    v4 = v1;
    if ( v3 > 1 )
    {
      v5 = v3;
      v6 = (__int64 *)&v26;
      v7 = 1;
      while ( 1 )
      {
        v24 = v6;
        v25 = 0x400000000LL;
        v9 = *(_QWORD *)(v4 + 8 * (v7 - *(unsigned int *)(v4 + 8)));
        if ( (unsigned __int8)(*(_BYTE *)v9 - 4) <= 0x1Eu )
        {
          v8 = *(_DWORD *)(v9 + 8);
          if ( !v8 )
            goto LABEL_5;
          v12 = *(_BYTE **)(v9 - 8LL * v8);
          if ( *v12 )
          {
            if ( v8 == 1 )
              goto LABEL_5;
            v12 = 0;
          }
          else if ( v8 == 1 )
          {
            goto LABEL_21;
          }
          v23 = v4;
          v13 = v6;
          v14 = v6;
          v22 = v5;
          v15 = 2;
          v16 = v12;
          v21 = v7;
          v17 = 0;
          v18 = *(_DWORD *)(v9 + 8);
          v19 = v9;
          v20 = *(_QWORD *)(v9 + 8 * (1LL - v8));
          while ( 1 )
          {
            v13[v17] = v20;
            v17 = (unsigned int)(v25 + 1);
            LODWORD(v25) = v25 + 1;
            if ( v18 <= (unsigned int)v15 )
              break;
            v20 = *(_QWORD *)(v19 + 8 * (v15 - *(unsigned int *)(v19 + 8)));
            if ( HIDWORD(v25) <= (unsigned int)v17 )
            {
              sub_16CD150((__int64)&v24, v14, 0, 8, v9, v2);
              v17 = (unsigned int)v25;
            }
            v13 = v24;
            ++v15;
          }
          v12 = v16;
          v5 = v22;
          v4 = v23;
          v6 = v14;
          v7 = v21;
          if ( v12 )
          {
LABEL_21:
            v9 = (__int64)v12;
LABEL_8:
            v11 = sub_161E970(v9);
            if ( (_DWORD)v25 == 1 )
              sub_1BF18E0((const char **)a1, v11, v10, *v24);
          }
          if ( v24 == v6 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v24);
          if ( v5 == ++v7 )
            return;
        }
        else
        {
          if ( !*(_BYTE *)v9 )
            goto LABEL_8;
LABEL_5:
          if ( v5 == ++v7 )
            return;
        }
      }
    }
  }
}
