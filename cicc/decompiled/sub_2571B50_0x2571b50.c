// Function: sub_2571B50
// Address: 0x2571b50
//
__int64 __fastcall sub_2571B50(_QWORD **a1, _BYTE *a2)
{
  __int64 v2; // rbx
  _DWORD *v3; // r12
  __int64 v4; // r8
  __int64 v5; // rcx
  __int64 v6; // r9
  int v7; // r14d
  __int64 *v8; // r11
  unsigned int v9; // eax
  __int64 *v10; // rdi
  _BYTE *v11; // rdx
  int v13; // esi
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r13
  int v17; // eax
  _BYTE *v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v19; // [rsp+8h] [rbp-38h] BYREF

  v2 = (*a1)[1];
  v3 = (_DWORD *)**a1;
  if ( !*a2 )
  {
    v4 = *(unsigned int *)(v2 + 144);
    v18 = a2;
    v5 = (__int64)a2;
    if ( (_DWORD)v4 )
    {
      v6 = *(_QWORD *)(v2 + 128);
      v7 = 1;
      v8 = 0;
      v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (__int64 *)(v6 + 8LL * v9);
      v11 = (_BYTE *)*v10;
      if ( a2 == (_BYTE *)*v10 )
        return 1;
      while ( v11 != (_BYTE *)-4096LL )
      {
        if ( v8 || v11 != (_BYTE *)-8192LL )
          v10 = v8;
        v9 = (v4 - 1) & (v7 + v9);
        v11 = *(_BYTE **)(v6 + 8LL * v9);
        if ( a2 == v11 )
          return 1;
        ++v7;
        v8 = v10;
        v10 = (__int64 *)(v6 + 8LL * v9);
      }
      v17 = *(_DWORD *)(v2 + 136);
      if ( !v8 )
        v8 = v10;
      ++*(_QWORD *)(v2 + 120);
      v14 = v17 + 1;
      v19 = v8;
      if ( 4 * (v17 + 1) < (unsigned int)(3 * v4) )
      {
        if ( (int)v4 - *(_DWORD *)(v2 + 140) - v14 > (unsigned int)v4 >> 3 )
          goto LABEL_13;
        v13 = v4;
LABEL_12:
        sub_A35F10(v2 + 120, v13);
        sub_A2AFD0(v2 + 120, (__int64 *)&v18, &v19);
        v5 = (__int64)v18;
        v8 = v19;
        v14 = *(_DWORD *)(v2 + 136) + 1;
LABEL_13:
        *(_DWORD *)(v2 + 136) = v14;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v2 + 140);
        *v8 = v5;
        v15 = *(unsigned int *)(v2 + 160);
        v16 = (__int64)v18;
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 164) )
        {
          sub_C8D5F0(v2 + 152, (const void *)(v2 + 168), v15 + 1, 8u, v4, v6);
          v15 = *(unsigned int *)(v2 + 160);
        }
        *(_QWORD *)(*(_QWORD *)(v2 + 152) + 8 * v15) = v16;
        ++*(_DWORD *)(v2 + 160);
        *v3 = 0;
        return 1;
      }
    }
    else
    {
      ++*(_QWORD *)(v2 + 120);
      v19 = 0;
    }
    v13 = 2 * v4;
    goto LABEL_12;
  }
  if ( !*(_BYTE *)(v2 + 168) )
    *v3 = 0;
  if ( !*(_BYTE *)(v2 + 169) )
    *v3 = 0;
  *(_WORD *)(v2 + 168) = 257;
  return 1;
}
