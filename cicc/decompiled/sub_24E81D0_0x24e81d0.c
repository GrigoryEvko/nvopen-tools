// Function: sub_24E81D0
// Address: 0x24e81d0
//
__int64 __fastcall sub_24E81D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // r12
  __int64 result; // rax
  __int64 *v4; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rax
  int v10; // edi
  _QWORD *v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 *v13; // rdi
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  __int64 v16; // r9
  unsigned int v17; // edi
  _QWORD *v18; // rdx
  __int64 v19; // r8
  int v20; // r11d
  int v21; // edi
  _QWORD *v22; // [rsp+0h] [rbp-80h]
  unsigned __int64 *v23; // [rsp+8h] [rbp-78h]
  _QWORD *v24; // [rsp+8h] [rbp-78h]
  _QWORD *v25; // [rsp+8h] [rbp-78h]
  _QWORD *v26; // [rsp+18h] [rbp-68h] BYREF
  void *v27; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v28[2]; // [rsp+28h] [rbp-58h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 24);
  v2 = *(__int64 **)(v1 + 8);
  result = *(unsigned int *)(v1 + 16);
  v4 = &v2[result];
  if ( v4 != v2 )
  {
    v6 = a1 + 200;
    while ( 1 )
    {
      v7 = *v2;
      v28[0] = 2;
      v28[1] = 0;
      v29 = v7;
      if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
        sub_BD73F0((__int64)v28);
      v8 = *(_DWORD *)(a1 + 224);
      v30 = v6;
      v27 = &unk_49DD7B0;
      if ( !v8 )
        break;
      v9 = v29;
      v16 = *(_QWORD *)(a1 + 208);
      v17 = (v8 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
      v19 = v18[3];
      if ( v29 == v19 )
      {
LABEL_20:
        v15 = v18 + 5;
        goto LABEL_21;
      }
      v20 = 1;
      v11 = 0;
      while ( v19 != -4096 )
      {
        if ( !v11 && v19 == -8192 )
          v11 = v18;
        v17 = (v8 - 1) & (v20 + v17);
        v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
        v19 = v18[3];
        if ( v29 == v19 )
          goto LABEL_20;
        ++v20;
      }
      v21 = *(_DWORD *)(a1 + 216);
      if ( !v11 )
        v11 = v18;
      ++*(_QWORD *)(a1 + 200);
      v10 = v21 + 1;
      v26 = v11;
      if ( 4 * v10 >= 3 * v8 )
        goto LABEL_8;
      if ( v8 - *(_DWORD *)(a1 + 220) - v10 <= v8 >> 3 )
        goto LABEL_9;
LABEL_10:
      *(_DWORD *)(a1 + 216) = v10;
      if ( v11[3] == -4096 )
      {
        v13 = v11 + 1;
        if ( v9 != -4096 )
          goto LABEL_15;
      }
      else
      {
        --*(_DWORD *)(a1 + 220);
        v12 = v11[3];
        if ( v9 != v12 )
        {
          v13 = v11 + 1;
          if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          {
            v22 = v11;
            v23 = v11 + 1;
            sub_BD60C0(v13);
            v9 = v29;
            v11 = v22;
            v13 = v23;
          }
LABEL_15:
          v11[3] = v9;
          if ( v9 == -4096 || v9 == 0 || v9 == -8192 )
          {
            v9 = v29;
          }
          else
          {
            v24 = v11;
            sub_BD6050(v13, v28[0] & 0xFFFFFFFFFFFFFFF8LL);
            v9 = v29;
            v11 = v24;
          }
        }
      }
      v14 = v30;
      v11[5] = 6;
      v11[6] = 0;
      v11[4] = v14;
      v15 = v11 + 5;
      v11[7] = 0;
LABEL_21:
      v27 = &unk_49DB368;
      if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
      {
        v25 = v15;
        sub_BD60C0(v28);
        v15 = v25;
      }
      ++v2;
      result = sub_24E76E0(v15[2], *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 288), 1, *(_QWORD *)(a1 + 288));
      if ( v4 == v2 )
        return result;
    }
    ++*(_QWORD *)(a1 + 200);
    v26 = 0;
LABEL_8:
    v8 *= 2;
LABEL_9:
    sub_CF32C0(v6, v8);
    sub_F9E960(v6, (__int64)&v27, &v26);
    v9 = v29;
    v10 = *(_DWORD *)(a1 + 216) + 1;
    v11 = v26;
    goto LABEL_10;
  }
  return result;
}
