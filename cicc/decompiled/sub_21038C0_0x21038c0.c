// Function: sub_21038C0
// Address: 0x21038c0
//
__int64 __fastcall sub_21038C0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  _QWORD *v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // edx
  __int16 v10; // r15
  _WORD *v11; // rdi
  __int16 *v12; // rdx
  unsigned __int16 v13; // r15
  _DWORD *v14; // r14
  __int16 *v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r9d
  __int16 v22; // ax
  unsigned int v23; // ecx
  _WORD *v24; // rax
  __int16 *v25; // rbx
  unsigned __int16 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  int v31; // r9d
  __int16 v32; // ax

  result = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)result )
  {
    if ( sub_21033E0(a1, a2, a3) )
    {
      return 3;
    }
    else if ( (unsigned __int8)sub_2103480(a1, a2, a3) )
    {
      return 2;
    }
    else
    {
      v7 = *(_QWORD **)(a1 + 232);
      if ( *(_QWORD *)(a2 + 104) )
      {
        if ( !v7 )
          BUG();
        v8 = v7[1] + 24LL * a3;
        v9 = *(_DWORD *)(v8 + 16);
        v10 = (v9 & 0xF) * a3;
        v11 = (_WORD *)(v7[7] + 2LL * (v9 >> 4));
        v12 = v11 + 1;
        v13 = *v11 + v10;
        v14 = (_DWORD *)(v7[8] + 4LL * *(unsigned __int16 *)(v8 + 20));
LABEL_7:
        v15 = v12;
        while ( v15 )
        {
          v16 = *(_QWORD *)(a2 + 104);
          if ( v16 )
          {
            while ( (*(_DWORD *)(v16 + 112) & *v14) == 0 )
            {
              v16 = *(_QWORD *)(v16 + 104);
              if ( !v16 )
                goto LABEL_14;
            }
            v17 = sub_2103840(a1, v16, v13);
            if ( (unsigned int)sub_20FD0B0(v17, 1u, v18, v19, v20, v21) )
              return 1;
          }
LABEL_14:
          v22 = *v15;
          ++v14;
          ++v15;
          v12 = 0;
          v13 += v22;
          if ( !v22 )
            goto LABEL_7;
        }
        return 0;
      }
      if ( !v7 )
        BUG();
      v23 = *(_DWORD *)(v7[1] + 24LL * a3 + 16);
      v24 = (_WORD *)(v7[7] + 2LL * (v23 >> 4));
      v25 = v24 + 1;
      v26 = *v24 + (v23 & 0xF) * a3;
LABEL_18:
      if ( !v25 )
        return 0;
      while ( 1 )
      {
        v27 = sub_2103840(a1, a2, v26);
        if ( (unsigned int)sub_20FD0B0(v27, 1u, v28, v29, v30, v31) )
          return 1;
        v32 = *v25++;
        if ( !v32 )
        {
          v25 = 0;
          goto LABEL_18;
        }
        v26 += v32;
        if ( !v25 )
          return 0;
      }
    }
  }
  return result;
}
