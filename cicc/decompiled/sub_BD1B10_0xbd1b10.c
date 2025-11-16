// Function: sub_BD1B10
// Address: 0xbd1b10
//
__int64 __fastcall sub_BD1B10(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *v3; // rbx
  int v4; // eax
  unsigned __int8 *v5; // rsi
  __int64 result; // rax
  __int64 *v7; // rbx
  __int64 *i; // r13
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r10d
  unsigned __int8 **v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  int v16; // eax
  int v17; // edx
  unsigned __int8 *v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned __int8 *v25; // rsi
  int v26; // r10d
  unsigned __int8 **v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  int v31; // r8d
  unsigned __int8 **v32; // rdi
  unsigned int v33; // r13d
  unsigned __int8 *v34; // rcx

  v3 = a2;
  v4 = *a2;
  if ( (_BYTE)v4 != 24 )
  {
LABEL_5:
    result = (unsigned int)(v4 - 4);
    if ( (unsigned __int8)result > 0x11u )
      return result;
    v10 = *(_DWORD *)(a1 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(a1 + 8);
      v12 = 1;
      v13 = 0;
      v14 = (v10 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v15 = (__int64 *)(v11 + 8LL * v14);
      result = *v15;
      if ( v3 == (unsigned __int8 *)*v15 )
        return result;
      while ( result != -4096 )
      {
        if ( result == -8192 && !v13 )
          v13 = (unsigned __int8 **)v15;
        v14 = (v10 - 1) & (v12 + v14);
        v15 = (__int64 *)(v11 + 8LL * v14);
        result = *v15;
        if ( (unsigned __int8 *)*v15 == v3 )
          return result;
        ++v12;
      }
      v16 = *(_DWORD *)(a1 + 16);
      if ( !v13 )
        v13 = (unsigned __int8 **)v15;
      ++*(_QWORD *)a1;
      v17 = v16 + 1;
      if ( 4 * (v16 + 1) < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(a1 + 20) - v17 > v10 >> 3 )
        {
LABEL_19:
          *(_DWORD *)(a1 + 16) = v17;
          if ( *v13 != (unsigned __int8 *)-4096LL )
            --*(_DWORD *)(a1 + 20);
          *v13 = v3;
          sub_BD0F10(a1, *((_QWORD *)v3 + 1));
          result = *v3;
          if ( (unsigned __int8)result <= 0x1Cu )
          {
            if ( (_BYTE)result == 5 && *((_WORD *)v3 + 1) == 34 )
            {
              v20 = sub_BB5290((__int64)v3);
              sub_BD0F10(a1, v20);
            }
            result = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
            v18 = &v3[-result];
            if ( (v3[7] & 0x40) != 0 )
            {
              v18 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
              v3 = &v18[result];
            }
            for ( ; v3 != v18; result = sub_BD1B10(a1, v19) )
            {
              v19 = *(_QWORD *)v18;
              v18 += 32;
            }
          }
          return result;
        }
        sub_BD14B0(a1, v10);
        v28 = *(_DWORD *)(a1 + 24);
        if ( v28 )
        {
          v29 = v28 - 1;
          v30 = *(_QWORD *)(a1 + 8);
          v31 = 1;
          v32 = 0;
          v33 = v29 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v13 = (unsigned __int8 **)(v30 + 8LL * v33);
          v34 = *v13;
          v17 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v13 != v3 )
          {
            while ( v34 != (unsigned __int8 *)-4096LL )
            {
              if ( v34 == (unsigned __int8 *)-8192LL && !v32 )
                v32 = v13;
              v33 = v29 & (v31 + v33);
              v13 = (unsigned __int8 **)(v30 + 8LL * v33);
              v34 = *v13;
              if ( *v13 == v3 )
                goto LABEL_19;
              ++v31;
            }
            if ( v32 )
              v13 = v32;
          }
          goto LABEL_19;
        }
LABEL_59:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_BD14B0(a1, 2 * v10);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v24 = (v21 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v13 = (unsigned __int8 **)(v23 + 8LL * v24);
      v25 = *v13;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      if ( v3 != *v13 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != (unsigned __int8 *)-4096LL )
        {
          if ( !v27 && v25 == (unsigned __int8 *)-8192LL )
            v27 = v13;
          v24 = v22 & (v26 + v24);
          v13 = (unsigned __int8 **)(v23 + 8LL * v24);
          v25 = *v13;
          if ( *v13 == v3 )
            goto LABEL_19;
          ++v26;
        }
        if ( v27 )
          v13 = v27;
      }
      goto LABEL_19;
    }
    goto LABEL_59;
  }
  while ( 1 )
  {
    v5 = (unsigned __int8 *)*((_QWORD *)v3 + 3);
    result = *v5;
    if ( (unsigned __int8)(result - 5) <= 0x1Fu )
      return sub_BD1850(a1, (__int64)v5);
    if ( (unsigned int)(unsigned __int8)result - 1 > 1 )
      break;
    v3 = (unsigned __int8 *)*((_QWORD *)v5 + 17);
    v4 = *v3;
    if ( (_BYTE)v4 != 24 )
      goto LABEL_5;
  }
  if ( (_BYTE)result == 4 )
  {
    v7 = (__int64 *)*((_QWORD *)v5 + 17);
    result = *((unsigned int *)v5 + 36);
    for ( i = &v7[result]; i != v7; result = sub_BD1B10(a1, *(_QWORD *)(v9 + 136)) )
      v9 = *v7++;
  }
  return result;
}
