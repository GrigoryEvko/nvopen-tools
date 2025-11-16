// Function: sub_21031A0
// Address: 0x21031a0
//
__int64 __fastcall sub_21031A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int *v7; // rax
  unsigned int v8; // esi
  _QWORD *v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rdi
  __int16 v14; // r13
  _WORD *v15; // r8
  unsigned __int16 *v16; // rdx
  unsigned __int16 v17; // r13
  _DWORD *v18; // r14
  unsigned __int16 *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // ecx
  __int16 v22; // dx
  __int64 v23; // rcx
  __int64 v24; // rbx
  unsigned __int16 v25; // r13

  v7 = (unsigned int *)(*(_QWORD *)(a1[31] + 264LL) + 4LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
  v8 = *v7;
  *v7 = 0;
  v9 = (_QWORD *)a1[29];
  if ( *(_QWORD *)(a2 + 104) )
  {
    if ( !v9 )
      BUG();
    v10 = v9[7];
    v11 = v9[1];
    result = v9[8];
    v13 = v11 + 24LL * v8;
    LODWORD(v11) = *(_DWORD *)(v13 + 16);
    v14 = v8 * (v11 & 0xF);
    v15 = (_WORD *)(v10 + 2LL * ((unsigned int)v11 >> 4));
    v16 = v15 + 1;
    v17 = *v15 + v14;
    v18 = (_DWORD *)(result + 4LL * *(unsigned __int16 *)(v13 + 20));
    while ( 1 )
    {
      v19 = v16;
      if ( !v16 )
        break;
      while ( 1 )
      {
        v20 = *(_QWORD *)(a2 + 104);
        if ( v20 )
        {
          while ( (*(_DWORD *)(v20 + 112) & *v18) == 0 )
          {
            v20 = *(_QWORD *)(v20 + 104);
            if ( !v20 )
              goto LABEL_10;
          }
          sub_20FD9D0((_DWORD *)(a1[48] + 216LL * v17), a2, v20, 27LL * v17, (__int64)v15);
        }
LABEL_10:
        result = *v19;
        ++v18;
        ++v19;
        v16 = 0;
        v17 += result;
        if ( !(_WORD)result )
          break;
        if ( !v19 )
          return result;
      }
    }
  }
  else
  {
    if ( !v9 )
      BUG();
    v21 = *(_DWORD *)(v9[1] + 24LL * v8 + 16);
    v22 = v21 & 0xF;
    v23 = v21 >> 4;
    result = v9[7] + 2 * v23;
    v24 = result + 2;
    v25 = *(_WORD *)result + v8 * v22;
    while ( v24 )
    {
      while ( 1 )
      {
        v24 += 2;
        sub_20FD9D0((_DWORD *)(a1[48] + 216LL * v25), a2, a2, v23, a5);
        result = *(unsigned __int16 *)(v24 - 2);
        if ( !(_WORD)result )
          break;
        v25 += result;
        if ( !v24 )
          return result;
      }
      v24 = 0;
    }
  }
  return result;
}
