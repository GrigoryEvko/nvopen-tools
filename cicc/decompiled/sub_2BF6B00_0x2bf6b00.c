// Function: sub_2BF6B00
// Address: 0x2bf6b00
//
void __fastcall sub_2BF6B00(__int64 a1, __int64 a2)
{
  char v4; // al
  bool v5; // zf
  unsigned int v6; // r14d
  _BYTE *v7; // r12
  unsigned int v8; // r13d
  _BYTE *v9; // rbx
  __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // esi
  __int64 v20; // r9
  int v21; // esi
  unsigned int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // rax
  _QWORD *v26; // rcx
  _BYTE *v27; // rsi
  _BYTE *v28; // rbx
  __int64 v29; // rdi
  _BYTE *v30; // rsi
  int v31; // eax
  int v32; // r11d
  _QWORD *v33; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v34; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h]
  _BYTE v36[112]; // [rsp+30h] [rbp-70h] BYREF

  v33 = *(_QWORD **)(a1 + 112);
  v34 = v36;
  v35 = 0x800000000LL;
  sub_2BF66C0((__int64)&v34, (__int64 *)&v33);
  v4 = *(_BYTE *)(a1 + 128);
  if ( !v4 )
  {
    v11 = *(_QWORD **)(a2 + 896);
    v12 = *(_QWORD *)(a2 + 928);
    v11[17] += 160LL;
    v13 = v11[7];
    v14 = (v13 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11[8] >= v14 + 160 && v13 )
      v11[7] = v14 + 160;
    else
      v14 = sub_9D1E70((__int64)(v11 + 7), 160, 160, 3);
    memset((void *)v14, 0, 0xA0u);
    *(_BYTE *)(v14 + 84) = 1;
    v15 = 0;
    *(_QWORD *)(v14 + 64) = v14 + 88;
    *(_QWORD *)(v14 + 72) = 8;
    *(_QWORD *)(a2 + 928) = v14;
    if ( *(_DWORD *)(a1 + 64) == 1 )
      v15 = **(_QWORD **)(a1 + 56);
    v33 = (_QWORD *)sub_2BF0520(v15);
    v16 = sub_2BF2B80(a2 + 120, (__int64 *)&v33);
    v17 = *(_QWORD *)(a2 + 896);
    v18 = *v16;
    v19 = *(_DWORD *)(v17 + 24);
    v20 = *(_QWORD *)(v17 + 8);
    if ( v19 )
    {
      v21 = v19 - 1;
      v22 = v21 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v18 == *v23 )
      {
LABEL_25:
        v25 = v23[1];
        v26 = *(_QWORD **)(a2 + 928);
        if ( v25 )
        {
          v33 = *(_QWORD **)(a2 + 928);
          *v26 = v25;
          v27 = *(_BYTE **)(v25 + 16);
          if ( v27 == *(_BYTE **)(v25 + 24) )
          {
            sub_D4C7F0(v25 + 8, v27, &v33);
          }
          else
          {
            if ( v27 )
            {
              *(_QWORD *)v27 = v33;
              v27 = *(_BYTE **)(v25 + 16);
            }
            *(_QWORD *)(v25 + 16) = v27 + 8;
          }
          goto LABEL_30;
        }
LABEL_36:
        v33 = v26;
        v30 = *(_BYTE **)(v17 + 40);
        if ( v30 == *(_BYTE **)(v17 + 48) )
        {
          sub_D4C7F0(v17 + 32, v30, &v33);
        }
        else
        {
          if ( v30 )
          {
            *(_QWORD *)v30 = v26;
            v30 = *(_BYTE **)(v17 + 40);
          }
          *(_QWORD *)(v17 + 40) = v30 + 8;
        }
LABEL_30:
        v28 = v34;
        v7 = &v34[8 * (unsigned int)v35];
        if ( v34 != v7 )
        {
          do
          {
            v29 = *((_QWORD *)v7 - 1);
            v7 -= 8;
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v29 + 16LL))(v29, a2);
          }
          while ( v28 != v7 );
          v7 = v34;
        }
        *(_QWORD *)(a2 + 928) = v12;
        if ( v7 == v36 )
          return;
LABEL_13:
        _libc_free((unsigned __int64)v7);
        return;
      }
      v31 = 1;
      while ( v24 != -4096 )
      {
        v32 = v31 + 1;
        v22 = v21 & (v31 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( v18 == *v23 )
          goto LABEL_25;
        v31 = v32;
      }
    }
    v26 = *(_QWORD **)(a2 + 928);
    goto LABEL_36;
  }
  v5 = *(_BYTE *)(a2 + 24) == 0;
  v6 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 16) = 0;
  *(_BYTE *)(a2 + 20) = 0;
  if ( v5 )
    *(_BYTE *)(a2 + 24) = 1;
  v7 = v34;
  v8 = 0;
  if ( v6 )
  {
    do
    {
      *(_DWORD *)(a2 + 16) = v8;
      *(_BYTE *)(a2 + 20) = 0;
      if ( !v4 )
        *(_BYTE *)(a2 + 24) = 1;
      v9 = &v7[8 * (unsigned int)v35];
      if ( v9 != v7 )
      {
        do
        {
          v10 = *((_QWORD *)v9 - 1);
          v9 -= 8;
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 16LL))(v10, a2);
        }
        while ( v9 != v7 );
        v7 = v34;
      }
      ++v8;
      v4 = *(_BYTE *)(a2 + 24);
    }
    while ( v8 < v6 );
    if ( !v4 )
    {
      if ( v7 == v36 )
        return;
      goto LABEL_13;
    }
  }
  else
  {
    v7 = v34;
  }
  *(_BYTE *)(a2 + 24) = 0;
  if ( v7 != v36 )
    goto LABEL_13;
}
