// Function: sub_2A69B70
// Address: 0x2a69b70
//
char __fastcall sub_2A69B70(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v6; // rcx
  int v7; // r11d
  __int64 *v8; // r10
  unsigned int v9; // edx
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  int v14; // esi
  int v15; // edx
  int v16; // eax
  __int64 v18; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v19; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a2 + 24);
  v3 = *a1;
  v18 = a2;
  v4 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned __int8)v4 <= 3u
    || (_BYTE)v4 == 5
    || (unsigned __int8)v4 <= 0x14u && (v13 = 1463376, _bittest64(&v13, v4)) )
  {
    v5 = *(_DWORD *)(v3 + 224);
    if ( v5 )
    {
      v6 = *(_QWORD *)(v3 + 208);
      v7 = 1;
      v8 = 0;
      v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (__int64 *)(v6 + 48LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
      {
LABEL_4:
        v12 = (__int64)(v10 + 1);
LABEL_5:
        LOBYTE(v4) = sub_2A624B0(v12, *(unsigned __int8 **)(a2 - 32), 0);
        return v4;
      }
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v8 )
          v8 = v10;
        v9 = (v5 - 1) & (v7 + v9);
        v10 = (__int64 *)(v6 + 48LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_4;
        ++v7;
      }
      v16 = *(_DWORD *)(v3 + 216);
      if ( !v8 )
        v8 = v10;
      ++*(_QWORD *)(v3 + 200);
      v15 = v16 + 1;
      v19 = v8;
      if ( 4 * (v16 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(v3 + 220) - v15 > v5 >> 3 )
          goto LABEL_14;
        v14 = v5;
LABEL_13:
        sub_2A698B0(v3 + 200, v14);
        sub_2A658B0(v3 + 200, &v18, &v19);
        a2 = v18;
        v8 = v19;
        v15 = *(_DWORD *)(v3 + 216) + 1;
LABEL_14:
        *(_DWORD *)(v3 + 216) = v15;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v3 + 220);
        *v8 = a2;
        v12 = (__int64)(v8 + 1);
        a2 = v18;
        *((_WORD *)v8 + 4) = 0;
        goto LABEL_5;
      }
    }
    else
    {
      ++*(_QWORD *)(v3 + 200);
      v19 = 0;
    }
    v14 = 2 * v5;
    goto LABEL_13;
  }
  return v4;
}
