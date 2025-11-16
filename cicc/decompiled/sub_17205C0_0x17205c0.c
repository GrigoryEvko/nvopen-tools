// Function: sub_17205C0
// Address: 0x17205c0
//
__int64 __fastcall sub_17205C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v5; // r9
  int v6; // r10d
  __int64 v7; // r8
  __int64 v8; // rdx
  unsigned int v9; // r14d
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  _QWORD *v12; // rax
  unsigned int v13; // esi
  int v14; // r15d
  _QWORD *v15; // r12
  int v16; // eax
  int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // eax
  int v20; // ecx
  int v21; // r10d
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // r14d
  _QWORD *v27; // rsi
  int v28; // r10d
  const void *v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  v30 = a1 + 2064;
  result = a1 + 16;
  v29 = (const void *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v12 = sub_1648700(v2);
      v13 = *(_DWORD *)(a1 + 2088);
      v14 = *(_DWORD *)(a1 + 8);
      v15 = v12;
      if ( !v13 )
        break;
      LODWORD(v5) = v13 - 1;
      v6 = 1;
      v7 = *(_QWORD *)(a1 + 2072);
      v8 = 0;
      v9 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
      v10 = (v13 - 1) & v9;
      result = v7 + 16LL * v10;
      v11 = *(_QWORD **)result;
      if ( v15 == *(_QWORD **)result )
      {
LABEL_4:
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return result;
      }
      else
      {
        while ( v11 != (_QWORD *)-8LL )
        {
          if ( v11 != (_QWORD *)-16LL || v8 )
            result = v8;
          v10 = v5 & (v6 + v10);
          v11 = *(_QWORD **)(v7 + 16LL * v10);
          if ( v15 == v11 )
            goto LABEL_4;
          ++v6;
          v8 = result;
          result = v7 + 16LL * v10;
        }
        if ( !v8 )
          v8 = result;
        v22 = *(_DWORD *)(a1 + 2080);
        ++*(_QWORD *)(a1 + 2064);
        v20 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a1 + 2084) - v20 <= v13 >> 3 )
          {
            sub_14672C0(v30, v13);
            v23 = *(_DWORD *)(a1 + 2088);
            if ( !v23 )
            {
LABEL_46:
              ++*(_DWORD *)(a1 + 2080);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(a1 + 2072);
            v7 = 0;
            v26 = v24 & v9;
            LODWORD(v5) = 1;
            v20 = *(_DWORD *)(a1 + 2080) + 1;
            v8 = v25 + 16LL * v26;
            v27 = *(_QWORD **)v8;
            if ( v15 != *(_QWORD **)v8 )
            {
              while ( v27 != (_QWORD *)-8LL )
              {
                if ( !v7 && v27 == (_QWORD *)-16LL )
                  v7 = v8;
                v28 = v5 + 1;
                LODWORD(v5) = v24 & (v26 + v5);
                v26 = v5;
                v8 = v25 + 16LL * (unsigned int)v5;
                v27 = *(_QWORD **)v8;
                if ( v15 == *(_QWORD **)v8 )
                  goto LABEL_23;
                LODWORD(v5) = v28;
              }
              if ( v7 )
                v8 = v7;
            }
          }
          goto LABEL_23;
        }
LABEL_7:
        sub_14672C0(v30, 2 * v13);
        v16 = *(_DWORD *)(a1 + 2088);
        if ( !v16 )
          goto LABEL_46;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 2072);
        v19 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v20 = *(_DWORD *)(a1 + 2080) + 1;
        v8 = v18 + 16LL * v19;
        v7 = *(_QWORD *)v8;
        if ( v15 != *(_QWORD **)v8 )
        {
          v21 = 1;
          v5 = 0;
          while ( v7 != -8 )
          {
            if ( v7 == -16 && !v5 )
              v5 = v8;
            v19 = v17 & (v21 + v19);
            v8 = v18 + 16LL * v19;
            v7 = *(_QWORD *)v8;
            if ( v15 == *(_QWORD **)v8 )
              goto LABEL_23;
            ++v21;
          }
          if ( v5 )
            v8 = v5;
        }
LABEL_23:
        *(_DWORD *)(a1 + 2080) = v20;
        if ( *(_QWORD *)v8 != -8 )
          --*(_DWORD *)(a1 + 2084);
        *(_QWORD *)v8 = v15;
        *(_DWORD *)(v8 + 8) = v14;
        result = *(unsigned int *)(a1 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, v29, 0, 8, v7, v5);
          result = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = v15;
        ++*(_DWORD *)(a1 + 8);
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 2064);
    goto LABEL_7;
  }
  return result;
}
