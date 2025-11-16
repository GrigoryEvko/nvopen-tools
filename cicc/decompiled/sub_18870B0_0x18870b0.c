// Function: sub_18870B0
// Address: 0x18870b0
//
void __fastcall sub_18870B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  const void *v4; // rbx
  _QWORD *v5; // rdi
  unsigned __int8 v6; // al
  char v7; // dl
  int v8; // edx
  __int64 v9; // rcx
  int v10; // esi
  unsigned int v11; // eax
  _QWORD *v12; // r10
  _QWORD *v13; // r8
  unsigned int v14; // esi
  unsigned int v15; // eax
  _QWORD *v16; // r9
  int v17; // ecx
  unsigned int v18; // edx
  int v19; // r8d
  __int64 v20; // rax
  int v21; // r11d
  _QWORD *v22; // [rsp+0h] [rbp-30h] BYREF
  _QWORD *v23; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = (const void *)(a2 + 96);
    while ( 1 )
    {
      v5 = sub_1648700(v2);
      v6 = *((_BYTE *)v5 + 16);
      if ( v6 != 3 )
      {
        v22 = 0;
        if ( v6 <= 0x10u )
          sub_18870B0(v5, a2);
        goto LABEL_7;
      }
      v7 = *(_BYTE *)(a2 + 8);
      v22 = v5;
      v8 = v7 & 1;
      if ( v8 )
      {
        v9 = a2 + 16;
        v10 = 7;
      }
      else
      {
        v14 = *(_DWORD *)(a2 + 24);
        v9 = *(_QWORD *)(a2 + 16);
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a2 + 8);
          ++*(_QWORD *)a2;
          v16 = 0;
          v17 = (v15 >> 1) + 1;
          goto LABEL_14;
        }
        v10 = v14 - 1;
      }
      v11 = v10 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v12 = (_QWORD *)(v9 + 8LL * v11);
      v13 = (_QWORD *)*v12;
      if ( v5 != (_QWORD *)*v12 )
        break;
LABEL_7:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return;
    }
    v21 = 1;
    v16 = 0;
    while ( v13 != (_QWORD *)-8LL )
    {
      if ( v16 || v13 != (_QWORD *)-16LL )
        v12 = v16;
      v11 = v10 & (v21 + v11);
      v13 = *(_QWORD **)(v9 + 8LL * v11);
      if ( v5 == v13 )
        goto LABEL_7;
      ++v21;
      v16 = v12;
      v12 = (_QWORD *)(v9 + 8LL * v11);
    }
    v15 = *(_DWORD *)(a2 + 8);
    if ( !v16 )
      v16 = v12;
    ++*(_QWORD *)a2;
    v17 = (v15 >> 1) + 1;
    if ( (_BYTE)v8 )
    {
      v18 = 24;
      v14 = 8;
    }
    else
    {
      v14 = *(_DWORD *)(a2 + 24);
LABEL_14:
      v18 = 3 * v14;
    }
    v19 = 4 * v17;
    if ( 4 * v17 >= v18 )
    {
      v14 *= 2;
    }
    else if ( v14 - *(_DWORD *)(a2 + 12) - v17 > v14 >> 3 )
    {
LABEL_17:
      *(_DWORD *)(a2 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
      if ( *v16 != -8 )
        --*(_DWORD *)(a2 + 12);
      *v16 = v5;
      v20 = *(unsigned int *)(a2 + 88);
      if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 92) )
      {
        sub_16CD150(a2 + 80, v4, 0, 8, v19, (int)v16);
        v20 = *(unsigned int *)(a2 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8 * v20) = v22;
      ++*(_DWORD *)(a2 + 88);
      goto LABEL_7;
    }
    sub_1886D00(a2, v14);
    sub_1882D40(a2, (__int64 *)&v22, &v23);
    v16 = v23;
    v5 = v22;
    v15 = *(_DWORD *)(a2 + 8);
    goto LABEL_17;
  }
}
