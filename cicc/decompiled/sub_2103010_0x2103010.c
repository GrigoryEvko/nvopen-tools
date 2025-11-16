// Function: sub_2103010
// Address: 0x2103010
//
__int64 __fastcall sub_2103010(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // r8
  _QWORD *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rcx
  __int16 v11; // r13
  _DWORD *v12; // r14
  _WORD *v13; // rsi
  unsigned __int16 *v14; // rdx
  unsigned __int16 v15; // r13
  unsigned __int16 *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // ecx
  __int16 v19; // dx
  __int64 v20; // rcx
  __int64 v21; // rbx
  unsigned __int16 v22; // r13

  sub_1F5BDB0(a1[31], *(_DWORD *)(a2 + 112), a3);
  v6 = (_QWORD *)a1[29];
  if ( *(_QWORD *)(a2 + 104) )
  {
    if ( !v6 )
      BUG();
    v7 = v6[7];
    v8 = v6[1];
    result = v6[8];
    v10 = v8 + 24LL * a3;
    LODWORD(v8) = *(_DWORD *)(v10 + 16);
    v11 = a3 * (v8 & 0xF);
    v12 = (_DWORD *)(result + 4LL * *(unsigned __int16 *)(v10 + 20));
    v13 = (_WORD *)(v7 + 2LL * ((unsigned int)v8 >> 4));
    v14 = v13 + 1;
    v15 = *v13 + v11;
    while ( 1 )
    {
      v16 = v14;
      if ( !v14 )
        break;
      while ( 1 )
      {
        v17 = *(_QWORD *)(a2 + 104);
        if ( v17 )
        {
          while ( (*(_DWORD *)(v17 + 112) & *v12) == 0 )
          {
            v17 = *(_QWORD *)(v17 + 104);
            if ( !v17 )
              goto LABEL_10;
          }
          sub_20FF4D0((_DWORD *)(a1[48] + 216LL * v15), a2, v17, 27LL * v15, v5);
        }
LABEL_10:
        result = *v16;
        ++v12;
        ++v16;
        v14 = 0;
        v15 += result;
        if ( !(_WORD)result )
          break;
        if ( !v16 )
          return result;
      }
    }
  }
  else
  {
    if ( !v6 )
      BUG();
    v18 = *(_DWORD *)(v6[1] + 24LL * a3 + 16);
    v19 = v18 & 0xF;
    v20 = v18 >> 4;
    result = v6[7] + 2 * v20;
    v21 = result + 2;
    v22 = *(_WORD *)result + a3 * v19;
    while ( v21 )
    {
      while ( 1 )
      {
        v21 += 2;
        sub_20FF4D0((_DWORD *)(a1[48] + 216LL * v22), a2, a2, v20, v5);
        result = *(unsigned __int16 *)(v21 - 2);
        if ( !(_WORD)result )
          break;
        v22 += result;
        if ( !v21 )
          return result;
      }
      v21 = 0;
    }
  }
  return result;
}
